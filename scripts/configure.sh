#!/usr/bin/env bash
# Configures the build tree.
#
# Usage: bash scripts/configure.sh [--build-type Release] [--arch "120"] [--no-tests] [--with-vllm]
#                                   [--stub] [--build-dir build] [--clean]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_TYPE="Release"
ARCH=""
BUILD_DIR="${PROJECT_ROOT}/build"
ENABLE_TESTS=ON
ENABLE_BENCHMARKS=ON
WITH_VLLM=OFF
INFRA_MODE=SOURCE
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --no-tests) ENABLE_TESTS=OFF; shift ;;
        --no-benchmarks) ENABLE_BENCHMARKS=OFF; shift ;;
        --with-vllm) WITH_VLLM=ON; shift ;;
        --stub) INFRA_MODE=STUB; shift ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

ARGS=(
    -S "${PROJECT_ROOT}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DSCI_ATTENTION_BUILD_TESTS="${ENABLE_TESTS}"
    -DSCI_ATTENTION_BUILD_BENCHMARKS="${ENABLE_BENCHMARKS}"
    -DSCI_ATTENTION_BUILD_VLLM_ADAPTER="${WITH_VLLM}"
    -DSCI_ATTENTION_INFRA_MODE="${INFRA_MODE}"
)
[[ -n "${ARCH}" ]] && ARGS+=(-DSCI_ATTENTION_ARCH="${ARCH}")
[[ "${CLEAN}" -eq 1 ]] && rm -rf "${BUILD_DIR}"

echo "[configure] cmake ${ARGS[*]}"
cmake "${ARGS[@]}"
echo "[configure] done: ${BUILD_DIR}"

