#!/usr/bin/env bash
# Runs a benchmark suite and stores the artefacts under benchmarks/results/<date>-<sha>/.
#
# Usage: bash scripts/bench.sh [--suite sanity|main|long] [--only <case>] [--runs N] [--warmup N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
SUITE="main"
ONLY=""
RUNS=100
WARMUP=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite) SUITE="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

SHA="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo nogit)"
OUT_DIR="${PROJECT_ROOT}/benchmarks/results/$(date +%F)-${SHA}"
mkdir -p "${OUT_DIR}"

echo "[bench] suite=${SUITE} out=${OUT_DIR}"
echo "[bench] GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm --format=csv,noheader

shopt -s nullglob
BENCHES=("${BUILD_DIR}"/benchmarks/benchmark_*)
shopt -u nullglob
if [[ ${#BENCHES[@]} -eq 0 ]]; then
    echo "[bench] no benchmark binaries found in ${BUILD_DIR}/benchmarks" >&2
    exit 2
fi

for bench in "${BENCHES[@]}"; do
    [[ -x "${bench}" ]] || continue
    name="$(basename "${bench}")"
    if [[ -n "${ONLY}" && "${name}" != *"${ONLY}"* ]]; then
        continue
    fi
    json="${OUT_DIR}/${name#benchmark_}.json"
    csv="${OUT_DIR}/${name#benchmark_}.csv"
    echo "[bench] ${name} -> ${json}"
    "${bench}" --suite "${SUITE}" --runs "${RUNS}" --warmup "${WARMUP}" \
        --json "${json}" --csv "${csv}"
done

echo "[bench] artefacts: ${OUT_DIR}"

