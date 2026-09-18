#!/usr/bin/env bash
# Removes build trees and profiling artefacts. Source code and stored results are kept unless
# --all is given (results are then removed as well).
#
# Usage: bash scripts/clean.sh [--all] [--build-dir build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all) ALL=1; shift ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

rm -rf "${BUILD_DIR}"
rm -rf "${PROJECT_ROOT}/profiling/reports"
rm -rf "${PROJECT_ROOT}/.triton_cache"
echo "[clean] removed ${BUILD_DIR}, profiling/reports, .triton_cache"

if [[ "${ALL}" -eq 1 ]]; then
    find "${PROJECT_ROOT}/benchmarks/results" -mindepth 1 -not -name '.gitkeep' -delete 2>/dev/null || true
    echo "[clean] removed stored benchmark results"
fi

