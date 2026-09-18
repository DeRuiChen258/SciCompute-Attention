#!/usr/bin/env python3
"""Minimal GGUF v3 reader (prompt §23.4).

The `gguf` and `llama_cpp` packages are not installed in the project environment, so this module
implements the container format directly:
  header   : magic "GGUF", version, tensor_count, metadata_kv_count
  metadata : typed key/value pairs (all scalar and array types in the spec)
  tensors  : name, dims, ggml type, offset

Dequantisation covers F32/F16 and Q8_0 (block = 2 B fp16 scale + 32 x int8, 34 B per 32 weights).
Any other type raises `UnsupportedGgmlType` with the numeric type in the message - silent skipping
is forbidden by the prompt.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8

GGML_TYPE_NAMES = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q8_0: "Q8_0"}

GGUF_METADATA_TYPES = {
    0: ("u8", 1), 1: ("i8", 1), 2: ("u16", 2), 3: ("i16", 2), 4: ("u32", 4), 5: ("i32", 4),
    6: ("f32", 4), 7: ("bool", 1), 8: ("str", None), 9: ("arr", None), 10: ("u64", 8),
    11: ("i64", 8), 12: ("f64", 8),
}


class UnsupportedGgmlType(RuntimeError):
    """Raised when a tensor uses a quantisation type this reader does not implement."""


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"UNKNOWN({self.ggml_type})")

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= int(dim)
        return total


@dataclass
class GgufFile:
    path: Path
    version: int
    metadata: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    tensor_data_offset: int = 0
    alignment: int = 32

    def find_tensor(self, name: str) -> TensorInfo | None:
        for info in self.tensors:
            if info.name == name:
                return info
        return None


class Reader:
    def __init__(self, path: Path):
        self.handle = open(path, "rb")
        self.path = path

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.handle.close()

    def read_scalar(self, type_id: int):
        fmt, size = GGUF_METADATA_TYPES[type_id]
        if fmt == "str":
            length = struct.unpack("<Q", self.handle.read(8))[0]
            return self.handle.read(length).decode("utf-8", errors="replace")
        if fmt == "bool":
            return bool(struct.unpack("<B", self.handle.read(1))[0])
        if fmt == "arr":
            raise ValueError("array must be read through read_value")
        raw = self.handle.read(size)
        if fmt == "f32":
            return struct.unpack("<f", raw)[0]
        if fmt == "f64":
            return struct.unpack("<d", raw)[0]
        return struct.unpack("<" + {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I",
                                    "i32": "i", "u64": "Q", "i64": "q"}[fmt], raw)[0]

    def read_value(self, type_id: int):
        if type_id != 9:
            return self.read_scalar(type_id)
        element_type = struct.unpack("<I", self.handle.read(4))[0]
        count = struct.unpack("<Q", self.handle.read(8))[0]
        if element_type == 8:  # array of strings
            return [self.read_scalar(8) for _ in range(count)]
        values = np.empty(count, dtype=object)
        for index in range(count):
            values[index] = self.read_scalar(element_type)
        return values


def read_gguf(path: Path, keep_arrays: bool = False) -> GgufFile:
    with Reader(path) as reader:
        magic = reader.handle.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file (magic={magic!r})")
        version = struct.unpack("<I", reader.handle.read(4))[0]
        tensor_count = struct.unpack("<Q", reader.handle.read(8))[0]
        metadata_count = struct.unpack("<Q", reader.handle.read(8))[0]

        gguf = GgufFile(path=path, version=version)
        for _ in range(metadata_count):
            key = reader.read_scalar(8)
            value_type = struct.unpack("<I", reader.handle.read(4))[0]
            value = reader.read_value(value_type)
            if isinstance(value, np.ndarray):
                if not keep_arrays:
                    value = f"<array len={len(value)} dtype={value.dtype}>"
                else:
                    value = value.tolist()
            gguf.metadata[key] = value

        for _ in range(tensor_count):
            name = reader.read_scalar(8)
            ndims = struct.unpack("<I", reader.handle.read(4))[0]
            dims = [struct.unpack("<Q", reader.handle.read(8))[0] for _ in range(ndims)]
            ggml_type = struct.unpack("<I", reader.handle.read(4))[0]
            offset = struct.unpack("<Q", reader.handle.read(8))[0]
            gguf.tensors.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type,
                                           offset=offset))

        alignment = int(gguf.metadata.get("general.alignment", 32))
        gguf.alignment = alignment
        position = reader.handle.tell()
        remainder = position % alignment
        gguf.tensor_data_offset = position + (alignment - remainder if remainder else 0)
        return gguf


def tensor_nbytes(info: TensorInfo) -> int:
    if info.ggml_type == GGML_F32:
        return info.num_elements * 4
    if info.ggml_type == GGML_F16:
        return info.num_elements * 2
    if info.ggml_type == GGML_Q8_0:
        if info.num_elements % 32 != 0:
            raise UnsupportedGgmlType(
                f"{info.name}: Q8_0 requires a multiple of 32 elements, got {info.num_elements}")
        return (info.num_elements // 32) * 34
    raise UnsupportedGgmlType(f"{info.name}: ggml type {info.ggml_type} is not implemented")


def load_tensor(path: Path, info: TensorInfo) -> np.ndarray:
    """Reads and dequantises one tensor. Returns a float32 array with the tensor's shape."""
    shape = list(reversed(info.dims))  # GGUF stores dims fastest-varying first
    with open(path, "rb") as handle:
        handle.seek(0, 2)
        file_size = handle.tell()
        gguf = read_gguf(path)
        if info.offset + tensor_nbytes(info) > file_size - gguf.tensor_data_offset:
            raise ValueError(f"{info.name}: tensor exceeds the file size")
        handle.seek(gguf.tensor_data_offset + info.offset)
        raw = handle.read(tensor_nbytes(info))

    if info.ggml_type == GGML_F32:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
    if info.ggml_type == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if info.ggml_type == GGML_Q8_0:
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
        scales = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
        quants = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
        return (scales * quants).reshape(-1).reshape(shape)
    raise UnsupportedGgmlType(f"{info.name}: ggml type {info.ggml_type} is not implemented")


def main() -> int:
    parser = argparse.ArgumentParser(description="GGUF v3 metadata / tensor tools")
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--meta", action="store_true", help="print metadata as JSON")
    parser.add_argument("--list-tensors", action="store_true", help="print the tensor table")
    parser.add_argument("--dump-tensor", default="", help="dequantise one tensor and print stats")
    parser.add_argument("--json", default="", help="write the selected output to a JSON file")
    args = parser.parse_args()

    gguf = read_gguf(args.gguf)
    payload = {"path": str(gguf.path), "version": gguf.version,
               "tensor_count": len(gguf.tensors), "metadata": gguf.metadata}
    if args.list_tensors:
        payload["tensors"] = [
            {"name": t.name, "dims": t.dims, "type": t.type_name, "bytes": tensor_nbytes(t),
             "offset": t.offset}
            for t in gguf.tensors
        ]
    if args.dump_tensor:
        info = gguf.find_tensor(args.dump_tensor)
        if info is None:
            print(f"tensor {args.dump_tensor!r} not found", file=sys.stderr)
            return 1
        values = load_tensor(args.gguf, info)
        payload["tensor"] = {"name": info.name, "type": info.type_name,
                             "shape": list(values.shape),
                             "min": float(values.min()), "max": float(values.max()),
                             "mean": float(values.mean())}
    if args.meta or args.list_tensors or args.dump_tensor:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps({"path": payload["path"], "version": gguf.version,
                          "tensor_count": len(gguf.tensors),
                          "metadata_keys": len(gguf.metadata)}, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"[gguf_reader] json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

