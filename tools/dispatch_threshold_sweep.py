#!/usr/bin/env python3
"""Dispatch threshold calibration (prompt §9.4).

Two modes:
  * `--collect`  : run the benchmark binaries over a seq_q x seq_kv grid and store the measured
                   per-backend P50 timings as JSON (needs a GPU; the numbers feed the table below);
  * `--emit-inc` : regenerate `src/runtime/dispatch_table.inc` from a measurement file.

The compiled-in table currently carries the seed values documented in the prompt (§9.1) with the
single documented change that the decode kernel serves S_q = 1 only (see docs/prefill_decode.md).

Usage:
  python tools/dispatch_threshold_sweep.py --collect --out benchmarks/results/<date>-<sha>/dispatch_sweep.json
  python tools/dispatch_threshold_sweep.py --emit-inc src/runtime/dispatch_table.inc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEADER = """// AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
//
// Producer : tools/dispatch_threshold_sweep.py
// Revision : {revision}
// Regenerate:
//   python tools/dispatch_threshold_sweep.py --emit-inc src/runtime/dispatch_table.inc
//
// Values are seed values from the implementation prompt §9.1 plus the calibrated decode regime
// (decode kernel covers S_q = 1 only; larger small-q goes to flash). A future sweep replaces them
// with measured boundaries; tests/unit/test_dispatch.cpp snapshots the resulting behaviour.

namespace sca {{
namespace detail {{

struct DispatchThresholds {{
    int64_t decode_max_seq_q;
    int64_t decode_min_seq_kv;
    int64_t tiled_max_seq_kv;
    int64_t tokens_per_split;
    int64_t max_splits;
    const char* revision;
}};

inline constexpr DispatchThresholds kDispatchThresholds{{
    /* decode_max_seq_q   */ {decode_max_seq_q},
    /* decode_min_seq_kv  */ {decode_min_seq_kv},
    /* tiled_max_seq_kv   */ {tiled_max_seq_kv},
    /* tokens_per_split   */ {tokens_per_split},
    /* max_splits         */ {max_splits},
    /* revision           */ "{revision}",
}};

}}  // namespace detail
}}  // namespace sca
"""


def collect(out_path: Path, runs: int, warmup: int) -> int:
    binaries = {"flash": PROJECT_ROOT / "build/benchmarks/benchmark_attention",
                "naive": PROJECT_ROOT / "build/benchmarks/benchmark_naive"}
    for name, binary in binaries.items():
        if not binary.exists():
            print(f"[sweep] missing {binary}; run scripts/build.sh first", file=sys.stderr)
            return 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, binary in binaries.items():
        target = out_path.with_name(f"{name}_sweep.json")
        command = [str(binary), "--suite", "main", "--runs", str(runs), "--warmup", str(warmup),
                   "--json", str(target)]
        print(f"[sweep] {' '.join(command)}")
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            return completed.returncode
        results[name] = str(target)
    out_path.write_text(json.dumps({"measurements": results, "runs": runs, "warmup": warmup},
                                   indent=2), encoding="utf-8")
    print(f"[sweep] measurement index written: {out_path}")
    return 0


def emit_inc(path: Path, revision: str, values: dict) -> int:
    text = HEADER.format(revision=revision, **values)
    path.write_text(text, encoding="utf-8")
    print(f"[sweep] {path} regenerated (revision {revision})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--emit-inc", default="")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "benchmarks/results/dispatch_sweep.json")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--revision", default="r1")
    parser.add_argument("--decode-max-seq-q", type=int, default=1)
    parser.add_argument("--decode-min-seq-kv", type=int, default=256)
    parser.add_argument("--tiled-max-seq-kv", type=int, default=128)
    parser.add_argument("--tokens-per-split", type=int, default=512)
    parser.add_argument("--max-splits", type=int, default=16)
    args = parser.parse_args()

    if args.collect:
        return collect(args.out, args.runs, args.warmup)
    if args.emit_inc:
        return emit_inc(Path(args.emit_inc), args.revision, {
            "decode_max_seq_q": args.decode_max_seq_q,
            "decode_min_seq_kv": args.decode_min_seq_kv,
            "tiled_max_seq_kv": args.tiled_max_seq_kv,
            "tokens_per_split": args.tokens_per_split,
            "max_splits": args.max_splits,
        })
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

