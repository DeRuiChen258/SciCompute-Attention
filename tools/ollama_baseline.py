#!/usr/bin/env python3
"""Collects a local Ollama service baseline (scoreboard for the attention layer).

Scope note (prompt §23.6): these numbers describe a *complete model service*
(tokenizer + weights + sampling + KV cache), not this project's attention kernels. They are the
reference that the rollout stub is compared against, and they must never be mixed with kernel
timings in the same table.

Only the Python standard library is used so the script runs in any interpreter.

Usage:
  python tools/ollama_baseline.py --check
  python tools/ollama_baseline.py --json docs/results/ollama_baseline.json
  python tools/ollama_baseline.py --prompt-file prompt.txt --max-tokens 256 --runs 3
  SCA_OLLAMA_HOST=127.0.0.1:11435 SCA_OLLAMA_MODEL=qwen3:4b-thinking-2507-q8_0 \
      python tools/ollama_baseline.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = os.environ.get("SCA_OLLAMA_HOST", "127.0.0.1:11435")
DEFAULT_MODEL = os.environ.get("SCA_OLLAMA_MODEL", "qwen3:4b-thinking-2507-q8_0")
DEFAULT_PROMPT = ("Explain in two sentences why FlashAttention reduces HBM traffic "
                  "compared with materialising the attention matrix.")


def api(host: str, path: str, payload: dict | None = None, timeout: float = 300.0):
    url = f"http://{host}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    return json.loads(body) if body.strip() else {}


def gpu_state() -> dict:
    if shutil.which("nvidia-smi") is None:
        return {"available": False}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
        name, used, total, util = [part.strip() for part in out.split(",")]
        return {"available": True, "name": name, "memory_used_mib": int(used.split()[0]),
                "memory_total_mib": int(total.split()[0]), "utilization_pct": int(util.split()[0])}
    except Exception as exc:  # pragma: no cover - depends on the host
        return {"available": False, "error": str(exc)}


def ms(value) -> float:
    return round(float(value or 0) / 1e6, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", default="")
    parser.add_argument("--check", action="store_true",
                        help="only verify that the service and model are reachable")
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    payload: dict = {
        "service": {"host": args.host, "model": args.model},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": gpu_state(),
        "scope_note": ("full model service baseline (Ollama); NOT this project's attention kernel "
                       "performance - see docs/model_integration.md §6"),
    }

    try:
        payload["service"]["version"] = api(args.host, "/api/version", timeout=5).get("version")
        tags = api(args.host, "/api/tags", timeout=5).get("models", [])
        payload["service"]["available_models"] = [m.get("name") for m in tags]
        loaded = api(args.host, "/api/ps", timeout=5).get("models", [])
        payload["service"]["loaded_models"] = [
            {"name": m.get("name"), "size_vram_mib": round(m.get("size_vram", 0) / 2**20, 1)}
            for m in loaded
        ]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        payload["service"]["reachable"] = False
        payload["service"]["error"] = str(exc)
        message = (
            f"[ollama_baseline] cannot reach http://{args.host} ({exc}).\n"
            "  Start a CUDA-backed server first, e.g.:\n"
            "    SCA_MODEL_DIR=<model dir> SCA_OLLAMA_MODEL=" + args.model +
            " bash scripts/serve_model.sh")
        print(message, file=sys.stderr)
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 2

    payload["service"]["reachable"] = True
    if args.model not in payload["service"]["available_models"]:
        print(f"[ollama_baseline] warning: model {args.model!r} is not in the local library "
              f"(available: {payload['service']['available_models']})", file=sys.stderr)

    print(f"[ollama_baseline] service http://{args.host} version "
          f"{payload['service']['version']} model={args.model}")
    if args.check:
        print(f"[ollama_baseline] loaded: {payload['service']['loaded_models']}")
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[ollama_baseline] json written: {args.json}")
        return 0

    runs = []
    for index in range(args.runs):
        request = {
            "model": args.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": args.num_ctx, "num_predict": args.max_tokens},
        }
        started = time.perf_counter()
        response = api(args.host, "/api/generate", payload=request, timeout=1800)
        wall_ms = (time.perf_counter() - started) * 1000.0
        eval_count = int(response.get("eval_count") or 0)
        eval_ms = ms(response.get("eval_duration"))
        runs.append({
            "run": index,
            "wall_ms": round(wall_ms, 3),
            "load_ms": ms(response.get("load_duration")),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_ms": ms(response.get("prompt_eval_duration")),
            "eval_count": eval_count,
            "eval_ms": eval_ms,
            "tokens_per_second": round(eval_count / (eval_ms / 1000.0), 2) if eval_ms > 0 else None,
            "total_ms": ms(response.get("total_duration")),
            "done_reason": response.get("done_reason"),
            "answer_preview": (response.get("response") or "")[:160].replace("\n", " "),
        })
        print(f"[ollama_baseline] run {index}: eval {eval_count} tok in {eval_ms:.1f} ms -> "
              f"{runs[-1]['tokens_per_second']} tok/s (done: {runs[-1]['done_reason']})")

    payload["runs"] = runs
    payload["summary"] = {
        "tokens_per_second_mean": round(
            sum(r["tokens_per_second"] or 0.0 for r in runs) / len(runs), 2),
        "prompt_eval_ms_mean": round(sum(r["prompt_eval_ms"] for r in runs) / len(runs), 3),
        "wall_ms_mean": round(sum(r["wall_ms"] for r in runs) / len(runs), 3),
    }
    payload["gpu_after"] = gpu_state()

    print(f"[ollama_baseline] mean {payload['summary']['tokens_per_second_mean']} tok/s, "
          f"VRAM now {payload['gpu_after'].get('memory_used_mib')} MiB")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ollama_baseline] json written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

