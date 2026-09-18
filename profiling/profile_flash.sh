#!/usr/bin/env bash
# ncu profile of the flash kernel (default shape: D=128, S=4096, fp16, causal).
#
# Usage: bash profiling/profile_flash.sh [--set roofline|full] [--target <binary>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SET="roofline"
TARGET="${PROJECT_ROOT}/build/benchmarks/benchmark_attention"
ARGS=(--suite sanity --runs 3 --warmup 1)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set) SET="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

command -v ncu >/dev/null || {
    echo "[profile_flash] ncu not found; install Nsight Compute to collect GPU counters" >&2
    exit 2
}
[[ -x "${TARGET}" ]] || { echo "[profile_flash] missing ${TARGET}; run scripts/build.sh" >&2; exit 2; }

mkdir -p "${SCRIPT_DIR}/reports"
OUT="${SCRIPT_DIR}/reports/flash_$(date +%F-%H%M).csv"
METRICS_FILE="${SCRIPT_DIR}/metrics/roofline_metrics.txt"
[[ "${SET}" == "full" ]] && METRICS_FILE="${SCRIPT_DIR}/metrics/fa_metrics.txt"

echo "[profile_flash] collecting into ${OUT} (metrics: ${METRICS_FILE})"
ncu --target-processes all \
    --kernel-name regex:"flash_fwd|decode_partial" \
    --metrics-file "${METRICS_FILE}" \
    --csv --page raw \
    --log-file "${OUT}" \
    "${TARGET}" "${ARGS[@]}"

echo "[profile_flash] summary:"
python3 "${SCRIPT_DIR}/analyze_ncu.py" "${OUT}" || true

