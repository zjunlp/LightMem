#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON="A1_JAKE"
MODEL=""
OUTPUT_DIR="output"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --person) PERSON="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --output-dir|--output-root) OUTPUT_DIR="$2"; shift 2 ;;
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

MODEL="${MODEL:-${OPENAI_MODEL:-gpt-5-mini}}"

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

EPISODIC_FILE="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}/30sec/episodic_triplets_30sec_${MODEL}.json"
SEMANTIC_DIR="${OUTPUT_DIR}/metadata/semantic_graph/${PERSON}"
CAND_FILE="${SEMANTIC_DIR}/semantic_candidates_${MODEL}.json"

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/egolife/semantic_graph/${PERSON}"
mkdir -p "$LOG_DIR" "$SEMANTIC_DIR"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="$MODEL"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo -e "${BLUE}Semantic Graph: extracting semantic triples...${NC}"
"$PYTHON_BIN" experiments/egolife/preprocess/semantic_graph/build_semantic_candidate.py \
    --episodic-file "$EPISODIC_FILE" \
    --output-dir "$SEMANTIC_DIR" \
    --model "$MODEL" \
    2>&1 | tee "$LOG_DIR/build_semantic_candidate_${TIMESTAMP}.log"

echo -e "${BLUE}Semantic Graph: consolidating semantic graph...${NC}"
"$PYTHON_BIN" experiments/egolife/preprocess/semantic_graph/consolidate_semantic_graph.py \
    --semantic-file "$CAND_FILE" \
    --output-dir "$SEMANTIC_DIR" \
    --model "$MODEL" \
    2>&1 | tee "$LOG_DIR/consolidate_semantic_graph_${TIMESTAMP}.log"

echo -e "${BLUE}Semantic graph construction completed for ${PERSON}.${NC}"
