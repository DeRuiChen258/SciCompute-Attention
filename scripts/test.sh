#!/usr/bin/env bash
# Builds and runs the C++ test suite (L1-L5) and optionally the Python tests.
#
# Usage: bash scripts/test.sh [--filter <regex>] [--with-python] [--sanitize] [--build-dir build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
FILTER=""
WITH_PYTHON=0
SANITIZE=0
PYTHON_BIN="${SCA_PYTHON:-/home/violet/Workspace/miniconda/envs/cuda_132/bin/python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter) FILTER="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --with-python) WITH_PYTHON=1; shift ;;
        --sanitize) SANITIZE=1; shift ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

bash "${SCRIPT_DIR}/build.sh" --build-dir "${BUILD_DIR}"

CTEST_ARGS=(--test-dir "${BUILD_DIR}" --output-on-failure)
if [[ "${SANITIZE}" -eq 1 ]]; then
    CTEST_ARGS+=(-L sanitize)
elif [[ -n "${FILTER}" ]]; then
    CTEST_ARGS+=(-R "${FILTER}")
else
    CTEST_ARGS+=(-LE sanitize)
fi

echo "[test] ctest ${CTEST_ARGS[*]}"
ctest "${CTEST_ARGS[@]}"

if [[ "${WITH_PYTHON}" -eq 1 ]]; then
    echo "[test] pytest (${PYTHON_BIN})"
    TRITON_CACHE_DIR="${PROJECT_ROOT}/.triton_cache" \
        "${PYTHON_BIN}" -m pytest "${PROJECT_ROOT}/python/tests" -q
fi

