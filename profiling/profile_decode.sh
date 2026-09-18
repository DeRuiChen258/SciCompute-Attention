#!/usr/bin/env bash
# ncu profile of the split-K decode kernel.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/profile_flash.sh" --set roofline "$@"

