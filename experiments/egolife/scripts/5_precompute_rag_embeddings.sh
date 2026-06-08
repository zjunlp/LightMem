#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON="A1_JAKE"
RETRIEVER_MODEL=""
OUTPUT_DIR="output"
GPU="0"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE=""
TEXT_EMBEDDING_MODEL_ARG=""
BATCH_SIZE=""
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --person) PERSON="$2"; shift 2 ;;
        --model|--retriever-model) RETRIEVER_MODEL="$2"; shift 2 ;;
        --output-dir|--output-root) OUTPUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --text-embedding-model) TEXT_EMBEDDING_MODEL_ARG="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --force-rebuild) FORCE_REBUILD=1; shift ;;
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

MEMORY_CELL_DIR="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}"
SEMANTIC_FILE="${OUTPUT_DIR}/metadata/semantic_graph/${PERSON}/semantic_graph_${RETRIEVER_MODEL}.json"

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/egolife/rag_embeddings/${PERSON}"
mkdir -p "$LOG_DIR"

CMD=(
    "$PYTHON_BIN" experiments/egolife/preprocess/precompute_rag_embeddings.py
    --person "$PERSON"
    --memory-cell-dir "$MEMORY_CELL_DIR"
    --semantic-file "$SEMANTIC_FILE"
    --text-embedding-model "$TEXT_EMBEDDING_MODEL"
    --gpu "$GPU"
)

if [[ -n "$BATCH_SIZE" ]]; then
    CMD+=(--batch-size "$BATCH_SIZE")
fi
if [[ "$FORCE_REBUILD" -eq 1 ]]; then
    CMD+=(--force-rebuild)
fi

echo -e "${BLUE}Precomputing EgoLife RAG embeddings for ${PERSON}...${NC}"
if [[ "${GPU,,}" == "cpu" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_DIR/precompute_rag_embeddings_${TIMESTAMP}.log"
else
    CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" 2>&1 | tee "$LOG_DIR/precompute_rag_embeddings_${TIMESTAMP}.log"
fi

echo -e "${BLUE}RAG embedding precompute completed for ${PERSON}.${NC}"
