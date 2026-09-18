#!/usr/bin/env python3
"""PyTorch SDPA reference benchmark (prompt §12.2).

Methodology mirrors the C++ benchmarks: warmup 20, 100 measured launches, CUDA-event timing,
P50/P90/P99 reported, JSON fields aligned with §12.4, and the SDPA backend is selected explicitly
(never "whatever torch picks").

Usage:
  python benchmarks/benchmark_sdpa.py --suite main --json <out>/sdpa.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = p * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def stats(samples):
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "std": statistics.pstdev(samples),
        "p50": percentile(samples, 0.50),
        "p90": percentile(samples, 0.90),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
    }


def attention_flops(b, hq, sq, skv, d, causal):
    if not causal:
        pairs = float(sq) * float(skv)
    else:
        diagonal = skv - sq
        pairs = sum(min(skv, i + diagonal + 1) for i in range(sq))
    return 4.0 * b * hq * d * pairs


def modeled_bytes(b, hq, hkv, sq, skv, d, elem):
    q = b * hq * sq * d * elem
    kv = b * hkv * skv * d * elem
    return q + 2 * kv + q


def run_case(shape, dtype, causal, backend, warmup, runs, seed):
    b, hq, hkv, sq, skv, d = (shape["B"], shape["Hq"], shape["Hkv"], shape["Sq"], shape["Skv"],
                              shape["D"])
    torch.manual_seed(seed)
    device = torch.device("cuda")
    elem = torch.tensor([], dtype=dtype).element_size()
    q = torch.randn(b, hq, sq, d, device=device, dtype=dtype) * 0.1
    k = torch.randn(b, hkv, skv, d, device=device, dtype=dtype) * 0.1
    v = torch.randn(b, hkv, skv, d, device=device, dtype=dtype) * 0.1
    k_expanded = k.repeat_interleave(hq // hkv, dim=1)
    v_expanded = v.repeat_interleave(hq // hkv, dim=1)

    backend_map = {
        "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        "mem_efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        "math": torch.nn.attention.SDPBackend.MATH,
    }
    dtype_name = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]
    case_id = (f"sdpa_{backend}_B{b}_Hq{hq}_Hkv{hkv}_Sq{sq}_Skv{skv}_D{d}_{dtype_name}"
               f"_causal{1 if causal else 0}")
    result = {
        "case_id": case_id,
        "impl": f"sdpa_{backend}",
        "params": {"B": b, "H": hq, "Hkv": hkv, "Sq": sq, "Skv": skv, "D": d,
                   "dtype": dtype_name, "dtype_bytes": elem, "causal": causal,
                   "sdpa_backend": backend},
        "latency_ms": {},
        "flops_effective": attention_flops(b, hq, sq, skv, d, causal),
        "bytes_moved_modeled": modeled_bytes(b, hq, hkv, sq, skv, d, elem),
        "registers_per_thread": -1, "smem_bytes": 0, "achieved_occupancy_pct": -1.0,
        "reference_max_abs_err": -1.0, "note": "",
    }
    try:
        with torch.nn.attention.sdpa_kernel(backend_map[backend]):
            for _ in range(warmup):
                F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=causal)
            torch.cuda.synchronize()
            samples = []
            for _ in range(runs):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=causal)
                stop.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(stop))
    except Exception as exc:  # backend availability depends on the machine
        result["skipped"] = f"{backend} unavailable: {exc}"
        return result

    result["latency_ms"] = stats(samples)
    result["tflops"] = result["flops_effective"] / (result["latency_ms"]["p50"] * 1e-3) / 1e12
    result["effective_bw_gbps"] = (result["bytes_moved_modeled"] /
                                   (result["latency_ms"]["p50"] * 1e-3) / 1e9)
    result["workspace_mb"] = 0.0
    result["peak_mem_mb"] = sum(t.numel() * t.element_size()
                                for t in (q, k_expanded, v_expanded)) / 2**20
    return result


def build_shapes(suite):
    if suite == "sanity":
        return [{"B": 1, "Hq": 8, "Hkv": 8, "Sq": 512, "Skv": 512, "D": 128},
                {"B": 1, "Hq": 8, "Hkv": 8, "Sq": 1, "Skv": 1024, "D": 128}]
    if suite == "long":
        return [{"B": 1, "Hq": 8, "Hkv": 8, "Sq": 8192, "Skv": 8192, "D": 128}]
    shapes = [{"B": 1, "Hq": 8, "Hkv": 8, "Sq": s, "Skv": s, "D": 128}
              for s in (512, 1024, 2048, 4096, 8192)]
    shapes += [{"B": 1, "Hq": 8, "Hkv": 8, "Sq": 1, "Skv": s, "D": 128}
               for s in (1024, 4096, 8192, 16384)]
    # Real Qwen3-4B-Thinking-2507 shape (GQA group 4).
    shapes += [{"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1024, "Skv": 1024, "D": 128},
               {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 2048, "Skv": 2048, "D": 128},
               {"B": 1, "Hq": 32, "Hkv": 8, "Sq": 1, "Skv": 4096, "D": 128}]
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="main", choices=["sanity", "main", "long"])
    parser.add_argument("--json", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[sdpa] CUDA is not available in this interpreter", file=sys.stderr)
        return 2

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    free_b, total_b = torch.cuda.mem_get_info()
    capability = torch.cuda.get_device_capability(0)
    header = {
        "benchmark_name": "benchmark_sdpa",
        "git_sha": subprocess.run(["git", "-C", project_root, "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or "nogit",
        "build_type": "Release",
        "gpu": torch.cuda.get_device_name(0),
        "compute_cap": f"{capability[0]}.{capability[1]}",
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip(),
        "cuda": torch.version.cuda or "unknown",
        "sm_clock_mhz": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "gpu_mem_free_mb": round(free_b / 2**20, 1),
        "gpu_mem_total_mb": round(total_b / 2**20, 1),
    }
    print(f"[sdpa] {header['gpu']} cc={header['compute_cap']} torch={torch.__version__} "
          f"free={header['gpu_mem_free_mb']:.0f} MiB")

    cases = []
    for shape in build_shapes(args.suite):
        prefill = shape["Sq"] > 1
        causal_options = (False, True) if prefill else (True,)
        for dtype in (torch.float16, torch.bfloat16):
            for causal in causal_options:
                for backend in ("flash", "mem_efficient", "math"):
                    case = run_case(shape, dtype, causal, backend, args.warmup, args.runs,
                                    args.seed)
                    cases.append(case)
                    if "skipped" in case:
                        print(f"[sdpa] {case['case_id']:<68} SKIPPED ({case['skipped']})")
                    else:
                        print(f"[sdpa] {case['case_id']:<68} "
                              f"p50={case['latency_ms']['p50']:9.4f} ms "
                              f"{case['tflops']:8.3f} TFLOPS")

    payload = dict(header)
    payload["cases"] = cases
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[sdpa] json written: {args.json}")
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as handle:
            handle.write("name,case_id,warmup,runs,mean_ms,median_ms,std_ms,p50_ms,p90_ms,p95_ms,"
                         "p99_ms,tflops,effective_bw_gbps,peak_mem_mb,unit\n")
            for case in cases:
                if "skipped" in case:
                    continue
                lat = case["latency_ms"]
                handle.write(f"{case['impl']},{case['case_id']},{args.warmup},{args.runs},"
                             f"{lat['mean']:.6f},{lat['median']:.6f},{lat['std']:.6f},"
                             f"{lat['p50']:.6f},{lat['p90']:.6f},{lat['p95']:.6f},{lat['p99']:.6f},"
                             f"{case['tflops']:.6f},{case['effective_bw_gbps']:.6f},"
                             f"{case['peak_mem_mb']:.3f},ms\n")
        print(f"[sdpa] csv written: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
