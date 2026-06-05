#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PERSON="A1_JAKE"
MODEL=""
NUM_KEYFRAMES=3
MAX_WORKERS=8
DAY=""
MAX_SYNC_FILES=""
BASE_DIR="data/EgoLife"
OUTPUT_DIR="output"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_FILE=""
SCALES=("30sec" "3min" "10min" "1h")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --person) PERSON="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --output-dir|--output-root) OUTPUT_DIR="$2"; shift 2 ;;
        --num-keyframes) NUM_KEYFRAMES="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --day) DAY="$2"; shift 2 ;;
        --max-sync-files) MAX_SYNC_FILES="$2"; shift 2 ;;
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

SYNC_DIR="${BASE_DIR}/EgoLifeCap/Sync"
FINECAP_FILE="${BASE_DIR}/EgoLifeCap/${PERSON}/${PERSON}.json"
EVENT_RECORD_FILE="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}/${PERSON}_record.json"
VIDEO_SEARCH_ROOT="$BASE_DIR"
FRAMES_DIR="tmp/egolife_keyframes_aug/${PERSON}"
TEMPORAL_CONTEXT_VIEWS_DIR="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}/temporal_context_views"

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR=".log/egolife/build_multimodal_memory_cell/${PERSON}"
mkdir -p "$LOG_DIR" "${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}"

export OPENAI_API_KEY="${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
export OPENAI_MODEL="$MODEL"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo -e "${BLUE}Multimodal Memory Cell: generating fine captions...${NC}"
"$PYTHON_BIN" experiments/egolife/preprocess/multimodal_memory_cell/generate_fine_caption.py \
    --sync-dir "$SYNC_DIR" \
    --output "$FINECAP_FILE" \
    2>&1 | tee "$LOG_DIR/generate_fine_caption_${TIMESTAMP}.log"

echo -e "${BLUE}Multimodal Memory Cell: building multimodal event records...${NC}"
CMD=(
    "$PYTHON_BIN" experiments/egolife/preprocess/multimodal_memory_cell/build_multimodal_event_record.py
    --input "$FINECAP_FILE"
    --output "$EVENT_RECORD_FILE"
    --sync-dir "$SYNC_DIR"
    --person "$PERSON"
    --video-search-root "$VIDEO_SEARCH_ROOT"
    --frames-dir "$FRAMES_DIR"
    --num-keyframes "$NUM_KEYFRAMES"
    --model "$MODEL"
    --max-workers "$MAX_WORKERS"
)
if [[ -n "$DAY" ]]; then
    CMD+=(--day "$DAY")
fi
if [[ -n "$MAX_SYNC_FILES" ]]; then
    CMD+=(--max-sync-files "$MAX_SYNC_FILES")
fi
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/build_multimodal_event_record_${TIMESTAMP}.log"

echo -e "${BLUE}Multimodal Memory Cell: generating temporal context views...${NC}"
"$PYTHON_BIN" experiments/egolife/preprocess/multimodal_memory_cell/generate_temporal_context_views.py \
    --person "$PERSON" \
    --json-path "$EVENT_RECORD_FILE" \
    --save-path "$TEMPORAL_CONTEXT_VIEWS_DIR" \
    2>&1 | tee "$LOG_DIR/generate_temporal_context_views_${TIMESTAMP}.log"

for SCALE in "${SCALES[@]}"; do
    echo -e "${BLUE}Multimodal Memory Cell: building episodic graph for ${SCALE}...${NC}"
    if [[ "$SCALE" == "30sec" ]]; then
        INPUT_FILE="$EVENT_RECORD_FILE"
    else
        INPUT_FILE="${TEMPORAL_CONTEXT_VIEWS_DIR}/temporal_context_views_${SCALE}.json"
    fi

    SCALE_OUTPUT_DIR="${OUTPUT_DIR}/metadata/multimodal_memory_cell/${PERSON}/${SCALE}"
    "$PYTHON_BIN" experiments/egolife/preprocess/multimodal_memory_cell/build_episodic_graph.py \
        --input-file "$INPUT_FILE" \
        --output-dir "$SCALE_OUTPUT_DIR" \
        --person "$PERSON" \
        --model "$MODEL" \
        --max-workers "$MAX_WORKERS" \
        2>&1 | tee "$LOG_DIR/build_episodic_graph_${SCALE}_${TIMESTAMP}.log"
done

echo -e "${BLUE}Multimodal memory cell construction completed for ${PERSON}.${NC}"
