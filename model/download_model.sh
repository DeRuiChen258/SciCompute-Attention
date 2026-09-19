#!/usr/bin/env bash
# Downloads the Qwen3-4B-Thinking-2507 Q8_0 weights into a self-contained model directory.
#
# The weights are intentionally *not* stored in this git repository (4.28 GB > GitHub's 100 MiB
# per-file limit). This script reproduces the exact artifact the benchmarks were measured on.
#
# Usage:
#   SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 bash model/download_model.sh
#
# Environment:
#   SCA_MODEL_DIR   target directory (required)
#   SCA_OLLAMA_BIN  ollama binary to use (default: `ollama` from PATH)
#   SCA_OLLAMA_HOST host:port of the temporary download server (default: 127.0.0.1:11436)
#   SCA_OLLAMA_MODEL model tag (default: qwen3:4b-thinking-2507-q8_0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCA_MODEL_DIR:-}"
OLLAMA_BIN="${SCA_OLLAMA_BIN:-ollama}"
HOST="${SCA_OLLAMA_HOST:-127.0.0.1:11436}"
TAG="${SCA_OLLAMA_MODEL:-qwen3:4b-thinking-2507-q8_0}"
EXPECTED_SHA256="012aa2736c32b7b74c3ca7b2da181b9e1d24a3973abf5510dee5590b27445440"
GGUF_NAME="Qwen3-4B-Thinking-2507-Q8_0.gguf"

[[ -n "${MODEL_DIR}" ]] || { echo "SCA_MODEL_DIR is required" >&2; exit 2; }
command -v "${OLLAMA_BIN}" >/dev/null || {
    echo "ollama binary not found: ${OLLAMA_BIN} (install Ollama first: https://ollama.com)" >&2
    exit 2
}
mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_DIR}/${GGUF_NAME}" ]]; then
    actual="$(sha256sum "${MODEL_DIR}/${GGUF_NAME}" | awk '{print $1}')"
    if [[ "${actual}" == "${EXPECTED_SHA256}" ]]; then
        echo "[download_model] ${GGUF_NAME} already present with the expected sha256"
        exit 0
    fi
    echo "[download_model] existing file has sha256 ${actual}; re-downloading" >&2
fi

echo "[download_model] starting a dedicated ollama server on ${HOST} (library=${MODEL_DIR}/.ollama)"
OLLAMA_MODELS="${MODEL_DIR}/.ollama" OLLAMA_HOST="${HOST}" \
    setsid nohup "${OLLAMA_BIN}" serve >"${MODEL_DIR}/download.log" 2>&1 < /dev/null &

for _ in $(seq 1 30); do
    sleep 1
    curl -sf --max-time 3 "http://${HOST}/api/version" >/dev/null 2>&1 && break
done
curl -sf --max-time 3 "http://${HOST}/api/version" >/dev/null 2>&1 || {
    echo "[download_model] server did not start; see ${MODEL_DIR}/download.log" >&2
    exit 3
}

echo "[download_model] pulling ${TAG} (≈ 4.3 GB)"
OLLAMA_HOST="${HOST}" "${OLLAMA_BIN}" pull "${TAG}"

# Hard-link the blob next to the model library so the directory is self-contained and moving it is
# enough to relocate the model (no double disk usage).
BLOB="${MODEL_DIR}/.ollama/blobs/sha256-${EXPECTED_SHA256}"
if [[ -f "${BLOB}" ]]; then
    ln -f "${BLOB}" "${MODEL_DIR}/${GGUF_NAME}"
    echo "[download_model] linked ${BLOB} -> ${MODEL_DIR}/${GGUF_NAME}"
else
    echo "[download_model] warning: expected blob ${BLOB} not found; verify the tag's digest" >&2
fi

if [[ -f "${MODEL_DIR}/${GGUF_NAME}" ]]; then
    actual="$(sha256sum "${MODEL_DIR}/${GGUF_NAME}" | awk '{print $1}')"
    echo "[download_model] sha256: ${actual}"
    [[ "${actual}" == "${EXPECTED_SHA256}" ]] || {
        echo "[download_model] sha256 mismatch (expected ${EXPECTED_SHA256})" >&2
        exit 4
    }
fi

echo "[download_model] done. Next:"
echo "  SCA_MODEL_DIR=${MODEL_DIR} bash ${SCRIPT_DIR}/../scripts/serve_model.sh"

