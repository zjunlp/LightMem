#!/usr/bin/env bash
# Start the LightMem web backend with one worker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_PYTHON="$REPO_ROOT/../envs/lightmem/bin/python"
if [ ! -x "$DEFAULT_PYTHON" ]; then
  DEFAULT_PYTHON="python3"
fi

PYTHON="${LIGHTMEM_PYTHON:-$DEFAULT_PYTHON}"
HOST="${LIGHTMEM_WEB_HOST:-127.0.0.1}"
PORT="${LIGHTMEM_WEB_PORT:-8077}"

MODEL_DIR="${LIGHTMEM_MODEL_DIR:-$REPO_ROOT/../model}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$MODEL_DIR/tiktoken_cache}"

cd "$SCRIPT_DIR"

exec "$PYTHON" -m uvicorn app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers 1 \
  "$@"
