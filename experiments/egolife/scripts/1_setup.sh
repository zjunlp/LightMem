#!/usr/bin/env bash

set -euo pipefail
trap 'echo ""; echo "Interrupted."; exit 130' INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATA_DIR="data/EgoLife"
DOWNLOAD_DATA=1
WITH_VISUAL=0
WITH_FLASH_ATTN=0
UV_BIN="${UV_BIN:-uv}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --skip-data|--no-download) DOWNLOAD_DATA=0; shift ;;
        --with-visual) WITH_VISUAL=1; shift ;;
        --with-flash-attn) WITH_FLASH_ATTN=1; shift ;;
        --uv) UV_BIN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

BLUE=$'\033[1;34m'
NC=$'\033[0m'
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT}/.log/egolife/setup"
mkdir -p "$LOG_DIR"

if ! command -v "$UV_BIN" >/dev/null 2>&1; then
    echo -e "${BLUE}uv could not be found, installing...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tee "$LOG_DIR/uv_install_${TIMESTAMP}.log"
    export PATH="$HOME/.local/bin:$PATH"
    UV_BIN="uv"
fi

SYNC_ARGS=()
if [[ "$WITH_VISUAL" -eq 1 ]]; then
    SYNC_ARGS+=(--extra visual)
fi
if [[ "$WITH_FLASH_ATTN" -eq 1 ]]; then
    SYNC_ARGS+=(--extra flash-attn)
fi

echo -e "${BLUE}Setting up EgoLife uv environment...${NC}"
(
    cd "$EXPERIMENT_DIR"
    "$UV_BIN" sync "${SYNC_ARGS[@]}"
) 2>&1 | tee "$LOG_DIR/uv_sync_${TIMESTAMP}.log"

if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
    if [[ "$DATA_DIR" = /* ]]; then
        TARGET_DATA_DIR="$DATA_DIR"
    else
        TARGET_DATA_DIR="${REPO_ROOT}/${DATA_DIR}"
    fi
    mkdir -p "$(dirname "$TARGET_DATA_DIR")"

    echo -e "${BLUE}Downloading EgoLife dataset to ${TARGET_DATA_DIR}...${NC}"
    (
        cd "$EXPERIMENT_DIR"
        "$UV_BIN" run hf download lmms-lab/EgoLife \
            --repo-type=dataset \
            --local-dir "$TARGET_DATA_DIR"
    ) 2>&1 | tee "$LOG_DIR/hf_download_${TIMESTAMP}.log"
fi

echo -e "${BLUE}Setup done. Activate with: source ${EXPERIMENT_DIR}/.venv/bin/activate${NC}"
