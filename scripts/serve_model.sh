#!/usr/bin/env bash
# Starts a local Ollama server backed by a self-contained model directory.
#
# All paths come from the environment, so the script works on any machine:
#   SCA_MODEL_DIR   directory holding the GGUF / .ollama store   (required)
#   SCA_OLLAMA_BIN  ollama binary with a CUDA runner             (default: `ollama` from PATH)
#   SCA_OLLAMA_HOST host:port for the server                     (default: 127.0.0.1:11435)
#   SCA_OLLAMA_MODEL model tag to warm up                        (optional)
#
# Usage:
#   SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 bash scripts/serve_model.sh
#   SCA_MODEL_DIR=... SCA_OLLAMA_MODEL=qwen3:4b-thinking-2507-q8_0 bash scripts/serve_model.sh --check
set -euo pipefail

MODEL_DIR="${SCA_MODEL_DIR:-}"
OLLAMA_BIN="${SCA_OLLAMA_BIN:-ollama}"
HOST="${SCA_OLLAMA_HOST:-127.0.0.1:11435}"
MODEL_TAG="${SCA_OLLAMA_MODEL:-}"
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) CHECK_ONLY=1; shift ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

api_url="http://${HOST}"

check_service() {
    curl -sf --max-time 5 "${api_url}/api/version" >/dev/null 2>&1
}

if check_service; then
    echo "[serve_model] ollama already listening on ${HOST}"
    curl -s --max-time 5 "${api_url}/api/tags" | head -c 400; echo
    [[ "${CHECK_ONLY}" -eq 1 ]] && exit 0
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    echo "[serve_model] service on ${HOST} is not reachable" >&2
    exit 2
fi

if [[ -z "${MODEL_DIR}" ]]; then
    echo "[serve_model] SCA_MODEL_DIR is required (directory holding the GGUF / .ollama store)" >&2
    exit 2
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[serve_model] model directory not found: ${MODEL_DIR}" >&2
    exit 2
fi
if ! command -v "${OLLAMA_BIN}" >/dev/null 2>&1; then
    echo "[serve_model] ollama binary not found: ${OLLAMA_BIN} (set SCA_OLLAMA_BIN)" >&2
    exit 2
fi

echo "[serve_model] starting ${OLLAMA_BIN} (models=${MODEL_DIR}, host=${HOST})"
OLLAMA_MODELS="${MODEL_DIR}/.ollama" OLLAMA_HOST="${HOST}" \
    setsid nohup "${OLLAMA_BIN}" serve >"${MODEL_DIR}/server.log" 2>&1 < /dev/null &

for _ in $(seq 1 20); do
    sleep 1
    if check_service; then
        echo "[serve_model] service ready on ${HOST} (log: ${MODEL_DIR}/server.log)"
        break
    fi
done

if ! check_service; then
    echo "[serve_model] service did not come up; last log lines:" >&2
    tail -n 20 "${MODEL_DIR}/server.log" >&2 || true
    exit 3
fi

if [[ -n "${MODEL_TAG}" ]]; then
    echo "[serve_model] warming up ${MODEL_TAG}"
    curl -s --max-time 300 "${api_url}/api/generate" \
        -d "{\"model\":\"${MODEL_TAG}\",\"prompt\":\"hi\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
        >/dev/null
    curl -s --max-time 5 "${api_url}/api/ps"; echo
fi

