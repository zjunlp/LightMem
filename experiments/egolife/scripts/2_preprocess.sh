#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON="A1_JAKE"
BASE_DIR="data/EgoLife"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --person) PERSON="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Python executable not found. Set PYTHON_BIN or pass --python."
        exit 1
    fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/egolife/preprocess/${PERSON}"
mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}"
export MAX_WORKERS_FILES="${MAX_WORKERS_FILES:-2}"
export MAX_WORKERS_LINES="${MAX_WORKERS_LINES:-8}"

TRANSLATED_DIR="${BASE_DIR}/EgoLifeCap/DenseCaption/translated/${PERSON}"
SYNC_DIR="${BASE_DIR}/EgoLifeCap/Sync"

echo -e "${BLUE}Translating DenseCaption for ${PERSON}...${NC}"
"$PYTHON_BIN" experiments/egolife/utils/translate_densecap.py \
    --person "$PERSON" \
    --input-path "${BASE_DIR}/EgoLifeCap/DenseCaption/${PERSON}" \
    --output-path "$TRANSLATED_DIR" \
    2>&1 | tee "$LOG_DIR/translate_densecap_${TIMESTAMP}.log"

echo -e "${BLUE}Generating Sync data for ${PERSON}...${NC}"
"$PYTHON_BIN" experiments/egolife/utils/generate_sync.py \
    --person "$PERSON" \
    --base-dir "$BASE_DIR" \
    --dense-caption-dir "${BASE_DIR}/EgoLifeCap/DenseCaption/${PERSON}" \
    --translated-dir "$TRANSLATED_DIR" \
    --sync-dir "$SYNC_DIR" \
    2>&1 | tee "$LOG_DIR/generate_sync_${TIMESTAMP}.log"

echo -e "${BLUE}Preprocess done. Output: ${SYNC_DIR}${NC}"
