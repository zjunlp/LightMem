"""Paths and runtime settings for the LightMem web backend."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = REPO_ROOT / "web"
FRONTEND_DIST = WEB_DIR / "frontend" / "dist"
MODEL_DIR = Path(
    os.environ.get("LIGHTMEM_MODEL_DIR", str(REPO_ROOT.parent / "model"))
)

DATA_DIR = Path(os.environ.get("LIGHTMEM_WEB_DATA", WEB_DIR / "data"))
PRESET_DIR = DATA_DIR / "presets"
QDRANT_DIR = DATA_DIR / "qdrant"
LOG_DIR = DATA_DIR / "logs"
UPLOAD_DIR = DATA_DIR / "uploads"
SECRETS_FILE = DATA_DIR / "secrets.json"

MAX_JOB_EVENTS = 4000
MAX_JOBS = 100

DEFAULT_COMPRESSOR_PATH = os.environ.get(
    "LIGHTMEM_COMPRESSOR_PATH",
    str(MODEL_DIR / "llmlingua-2-bert-base-multilingual-cased-meetingbank"),
)
DEFAULT_EMBEDDER_PATH = os.environ.get(
    "LIGHTMEM_EMBEDDER_PATH", str(MODEL_DIR / "all-MiniLM-L6-v2")
)

TIKTOKEN_CACHE_DIR = os.environ.get(
    "TIKTOKEN_CACHE_DIR", str(MODEL_DIR / "tiktoken_cache")
)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", TIKTOKEN_CACHE_DIR)


def ensure_dirs() -> None:
    for d in (DATA_DIR, PRESET_DIR, QDRANT_DIR, LOG_DIR, UPLOAD_DIR):
        d.mkdir(parents=True, exist_ok=True)
