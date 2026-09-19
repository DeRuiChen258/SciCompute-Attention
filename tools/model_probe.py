#!/usr/bin/env python3
"""Qwen3-4B-Thinking-2507-Q8 probe (prompt §23.4).

Reads the GGUF metadata and emits docs/results/qwen3_4b_shapes.json with
  * attention geometry (layers / heads / kv heads / head dim / hidden / rope base),
  * the QKV/O projection tensor table with byte sizes,
  * the KV cache budget table for the layouts this project implements,
  * the Ollama service status (version + resident models + VRAM).

Usage:
  python tools/model_probe.py [--gguf <path>] [--out docs/results/qwen3_4b_shapes.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import read_gguf, tensor_nbytes  # noqa: E402

# Paths are configuration, not constants: point the probe at any model directory via
#   SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 python tools/model_probe.py
_MODEL_DIR_ENV = os.environ.get("SCA_MODEL_DIR", "")
DEFAULT_GGUF = Path(os.environ.get(
    "SCA_GGUF_PATH",
    str(Path(_MODEL_DIR_ENV) / "Qwen3-4B-Thinking-2507-Q8_0.gguf") if _MODEL_DIR_ENV else
    "Qwen3-4B-Thinking-2507-Q8_0.gguf"))
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "docs/results/qwen3_4b_shapes.json"
OLLAMA_HOST = os.environ.get("SCA_OLLAMA_URL",
                             "http://" + os.environ.get("SCA_OLLAMA_HOST", "127.0.0.1:11435"))


def display_path(path: Path) -> str:
    """Renders a path with the configured model directory replaced by its variable name."""
    text = str(path)
    if _MODEL_DIR_ENV and text.startswith(_MODEL_DIR_ENV):
        return text.replace(_MODEL_DIR_ENV, "$SCA_MODEL_DIR", 1)
    return text


def ollama_status() -> dict:
    status = {"host": OLLAMA_HOST, "reachable": False, "version": None, "loaded_models": [],
              "vram_mb": None}
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/version", timeout=3) as response:
            status["version"] = json.load(response).get("version")
        status["reachable"] = True
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/ps", timeout=3) as response:
            models = json.load(response).get("models", [])
            status["loaded_models"] = [m.get("name") for m in models]
            status["vram_bytes"] = sum(m.get("size_vram", 0) for m in models)
            status["vram_mb"] = round(status["vram_bytes"] / 2**20, 1)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        status["error"] = str(exc)
    return status


def gpu_state() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
        name, used, total = [part.strip() for part in out.split(",")]
        return {"name": name, "memory_used_mib": int(used.split()[0]),
                "memory_total_mib": int(total.split()[0])}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def kv_budget(num_layers: int, kv_heads: int, head_dim: int, dtype_bytes: int,
              block_size: int) -> dict:
    per_token = 2 * num_layers * kv_heads * head_dim * dtype_bytes
    per_block = per_token * block_size
    return {
        "bytes_per_token": per_token,
        "kib_per_token": round(per_token / 1024, 1),
        "block_size": block_size,
        "bytes_per_block": per_block,
        "mib_per_block": round(per_block / 2**20, 3),
        "contexts": [
            {"context": ctx, "bytes": per_token * ctx, "mib": round(per_token * ctx / 2**20, 1),
             "note": note}
            for ctx, note in ((512, "fits"), (1024, "fits"), (2048, "fits"),
                              (4096, "fits after stopping Ollama"),
                              (8192, "needs Ollama stopped, batch=1"),
                              (16384, "stop Ollama + strict budget"),
                              (32768, "not feasible next to the 4B weights"),
                              (262144, "design ceiling only"))
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    if not args.gguf.exists():
        print(f"[model_probe] GGUF not found: {args.gguf}", file=sys.stderr)
        return 2

    gguf = read_gguf(args.gguf)
    meta = gguf.metadata
    layers = int(meta["qwen3.block_count"])
    heads = int(meta["qwen3.attention.head_count"])
    kv_heads = int(meta["qwen3.attention.head_count_kv"])
    key_length = int(meta["qwen3.attention.key_length"])
    value_length = int(meta["qwen3.attention.value_length"])
    hidden = int(meta["qwen3.embedding_length"])
    context = int(meta["qwen3.context_length"])
    rope_base = float(meta["qwen3.rope.freq_base"])

    projections = []
    for info in gguf.tensors:
        if any(tag in info.name for tag in ("attn_q.", "attn_k.", "attn_v.", "attn_output.")):
            projections.append({"name": info.name, "dims": info.dims, "type": info.type_name,
                                "elements": info.num_elements, "bytes": tensor_nbytes(info)})
    projection_bytes = sum(entry["bytes"] for entry in projections)
    q_projection = next((p for p in projections if p["name"].endswith("attn_q.weight")), None)

    payload = {
        "model": {
            "path": display_path(args.gguf),
            "file_size_bytes": args.gguf.stat().st_size,
            "gguf_version": gguf.version,
            "tensor_count": len(gguf.tensors),
            "architecture": meta.get("general.architecture"),
            "file_type": meta.get("general.file_type"),
            "parameter_count": meta.get("general.parameter_count"),
        },
        "attention": {
            "num_layers": layers,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "group_size": heads // kv_heads,
            "key_length": key_length,
            "value_length": value_length,
            "head_dim": key_length,
            "hidden_size": hidden,
            "context_length": context,
            "rope_freq_base": rope_base,
            "causal": True,
            "projection_elements": q_projection["elements"] if q_projection else None,
        },
        "attention_projections": {
            "count": len(projections),
            "total_bytes": projection_bytes,
            "total_mib": round(projection_bytes / 2**20, 1),
            "per_layer_bytes": round(projection_bytes / layers, 1),
            "tensors": projections,
        },
        "kv_budget_fp16": kv_budget(layers, kv_heads, key_length, 2, args.block_size),
        "kv_budget_fp32": kv_budget(layers, kv_heads, key_length, 4, args.block_size),
        "benchmark_shapes": {
            "source": "real-model",
            "prefill": [f"B=1,Hq={heads},Hkv={kv_heads},S={s},D={key_length},causal=true"
                        for s in (512, 1024, 2048, 4096)],
            "decode": [f"B=1,Hq={heads},Hkv={kv_heads},S_q=1,S_kv={s},D={key_length},causal=true"
                       for s in (1024, 4096, 8192)],
        },
        "environment": {"gpu": gpu_state(), "ollama": ollama_status(),
                        "ollama_baseline_enabled": True},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[model_probe] layers={layers} Hq={heads} Hkv={kv_heads} D={key_length} "
          f"hidden={hidden} rope_base={rope_base:g}")
    print(f"[model_probe] attention projections: {len(projections)} tensors, "
          f"{payload['attention_projections']['total_mib']} MiB")
    print(f"[model_probe] KV: {payload['kv_budget_fp16']['kib_per_token']} KiB/token (fp16), "
          f"{payload['kv_budget_fp16']['mib_per_block']} MiB per {args.block_size}-token block")
    print(f"[model_probe] Ollama reachable={payload['environment']['ollama']['reachable']} "
          f"vram_mb={payload['environment']['ollama'].get('vram_mb')}")
    print(f"[model_probe] json written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
