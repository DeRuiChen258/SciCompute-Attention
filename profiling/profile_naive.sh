#!/usr/bin/env bash
# ncu profile of the naive reference kernels (small shapes only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

command -v ncu >/dev/null || { echo "[profile_naive] ncu not found" >&2; exit 2; }
mkdir -p "${SCRIPT_DIR}/reports"
OUT="${SCRIPT_DIR}/reports/naive_$(date +%F-%H%M).csv"

ncu --target-processes all \
    --kernel-name regex:"naive_" \
    --metrics-file "${SCRIPT_DIR}/metrics/roofline_metrics.txt" \
    --csv --page raw --log-file "${OUT}" \
    "${PROJECT_ROOT}/build/benchmarks/benchmark_naive" --suite sanity --runs 2 --warmup 1

python3 "${SCRIPT_DIR}/analyze_ncu.py" "${OUT}" || true

