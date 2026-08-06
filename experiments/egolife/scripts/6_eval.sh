#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON="A1_JAKE"
RETRIEVER_MODEL=""
RESPOND_MODEL="gpt-5"
EPISODIC_TOP_K="5"
SEMANTIC_TOP_K="8"
VISUAL_TOP_K="3"
NUM_WORKERS="8"
MAX_ROUNDS="3"
MAX_ERRORS="3"
OUTPUT_DIR="output"
DATA_DIR="data/EgoLife"
GPU_LIST="0,1"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE=""
TEXT_EMBEDDING_MODEL_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --person) PERSON="$2"; shift 2 ;;
        --retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
        --respond-model) RESPOND_MODEL="$2"; shift 2 ;;
        --episodic-top-k) EPISODIC_TOP_K="$2"; shift 2 ;;
        --semantic-top-k) SEMANTIC_TOP_K="$2"; shift 2 ;;
        --visual-top-k) VISUAL_TOP_K="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --max-errors) MAX_ERRORS="$2"; shift 2 ;;
        --output-dir|--output-root) OUTPUT_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --text-embedding-model) TEXT_EMBEDDING_MODEL_ARG="$2"; shift 2 ;;
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

RETRIEVER_MODEL="${RETRIEVER_MODEL:-${OPENAI_MODEL:-gpt-5-mini}}"
TEXT_EMBEDDING_MODEL="${TEXT_EMBEDDING_MODEL_ARG:-${EM2MEM_TEXT_EMBEDDING_MODEL:-${TEXT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}}}"

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

MULTIMODAL_MEMORY_CELL_DIR="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}"
SEMANTIC_GRAPH_DIR="${OUTPUT_DIR}/metadata/semantic_graph/${PERSON}"
VISUAL_DIR="${OUTPUT_DIR}/metadata/visual_memory/${PERSON}"
VISUAL_EVIDENCE_FILE="${MULTIMODAL_MEMORY_CELL_DIR}/${PERSON}_record.json"

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/egolife/eval/${PERSON}"
mkdir -p "$LOG_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="$RETRIEVER_MODEL"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo -e "${BLUE}Evaluating EgoLife ${PERSON}...${NC}"
"$PYTHON_BIN" experiments/egolife/eval/eval.py \
    --person "$PERSON" \
    --retriever-model "$RETRIEVER_MODEL" \
    --respond-model "$RESPOND_MODEL" \
    --episodic-top-k "$EPISODIC_TOP_K" \
    --semantic-top-k "$SEMANTIC_TOP_K" \
    --visual-top-k "$VISUAL_TOP_K" \
    --num-workers "$NUM_WORKERS" \
    --max-rounds "$MAX_ROUNDS" \
    --max-errors "$MAX_ERRORS" \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --gpu-list "$GPU_LIST" \
    --text-embedding-model "$TEXT_EMBEDDING_MODEL" \
    --memory-cell-dir "$MULTIMODAL_MEMORY_CELL_DIR" \
    --semantic-graph-dir "$SEMANTIC_GRAPH_DIR" \
    --visual-dir "$VISUAL_DIR" \
    --visual-evidence-file "$VISUAL_EVIDENCE_FILE" \
    2>&1 | tee "${LOG_DIR}/eval_${PERSON}_${TIMESTAMP}.log"

echo -e "${BLUE}Evaluation completed for ${PERSON}.${NC}"
