#!/usr/bin/env bash
# Builds the configured tree.
#
# Usage: bash scripts/build.sh [--target <name>] [-j N] [--build-dir build]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
JOBS="$(nproc)"
TARGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -j) JOBS="$2"; shift 2 ;;
        -j*) JOBS="${1#-j}"; shift ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "[build] no cache in ${BUILD_DIR}; run scripts/configure.sh first" >&2
    exit 2
fi

ARGS=(--build "${BUILD_DIR}" -j "${JOBS}")
[[ -n "${TARGET}" ]] && ARGS+=(--target "${TARGET}")

echo "[build] cmake ${ARGS[*]}"
cmake "${ARGS[@]}"
