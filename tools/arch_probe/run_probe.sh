#!/usr/bin/env bash
# Runs the full ISA probe suite and records raw evidence.
#
# Usage: bash tools/arch_probe/run_probe.sh [--out <dir>]
# Exit codes: 0 = every mandatory probe passed, 1 = a mandatory probe failed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${PROJECT_ROOT}/profiling/reports"
ARCH="${SCI_ATTENTION_ARCH:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "${OUT_DIR}"

if [[ -z "${ARCH}" ]]; then
    # Detect the compute capability of device 0 without hardcoding sm_120.
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
fi
echo "[probe] target arch: sm_${ARCH}"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "${BUILD_DIR}"' EXIT

nvcc -std=c++17 -O2 -arch="sm_${ARCH}" "${SCRIPT_DIR}/arch_probe.cu" -lcuda \
     -o "${BUILD_DIR}/arch_probe" 2>&1 | tee "${OUT_DIR}/arch_probe_compile.log"
echo "[probe] arch_probe compiled (exit ${PIPESTATUS[0]})"

set +e
"${BUILD_DIR}/arch_probe" --json > "${OUT_DIR}/arch_probe.json" 2>&1
rc=$?
"${BUILD_DIR}/arch_probe" | tee "${OUT_DIR}/arch_probe.log"
set -e
echo "[probe] arch_probe exit code: ${rc}"

echo "[probe] compiling wgmma_probe.cu (expected to FAIL on sm_120) ..."
set +e
nvcc -std=c++17 -O2 -arch="sm_${ARCH}" -c "${SCRIPT_DIR}/wgmma_probe.cu" \
     -o "${BUILD_DIR}/wgmma_probe.o" > "${OUT_DIR}/wgmma_probe.log" 2>&1
wgmma_rc=$?
set -e
if [[ ${wgmma_rc} -ne 0 ]]; then
    echo "[probe] wgmma: FAIL (expected) -- ptxas diagnostic:"
    grep -m3 -E "error|not supported" "${OUT_DIR}/wgmma_probe.log" || true
else
    echo "[probe] wgmma: PASS (unexpected on this toolchain) -- update docs/env_report.md"
fi

echo "[probe] evidence written to ${OUT_DIR}"
exit ${rc}

