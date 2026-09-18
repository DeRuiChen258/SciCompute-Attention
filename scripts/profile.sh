#!/usr/bin/env bash
# Thin wrapper around profiling/profile_*.sh.
#
# Usage: bash scripts/profile.sh --kernel flash|decode|naive|triton|vllm [--set full|roofline]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KERNEL="flash"
SET="roofline"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL="$2"; shift 2 ;;
        --set) SET="$2"; shift 2 ;;
        -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

SCRIPT="${PROJECT_ROOT}/profiling/profile_${KERNEL}.sh"
if [[ ! -f "${SCRIPT}" ]]; then
    echo "[profile] ${SCRIPT} does not exist yet (Phase 6)" >&2
    exit 2
fi
bash "${SCRIPT}" --set "${SET}"

