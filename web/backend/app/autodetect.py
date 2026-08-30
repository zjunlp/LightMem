"""Environment detection for local models, GPUs, Ollama, and tiktoken."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import DEFAULT_COMPRESSOR_PATH, DEFAULT_EMBEDDER_PATH, MODEL_DIR

SEARCH_ROOTS = [
    Path(p)
    for p in os.environ.get(
        "LIGHTMEM_MODEL_ROOTS",
        os.pathsep.join([str(MODEL_DIR), os.path.expanduser("~/.cache/huggingface/hub")]),
    ).split(os.pathsep)
    if p
]

_COMPRESSOR_HINTS = ("llmlingua",)
_EMBEDDER_HINTS = ("minilm", "bge", "gte", "e5", "embed", "sentence", "paraphrase", "qwen3-embedding")


def _is_model_dir(path: Path) -> bool:
    return (path / "config.json").is_file()


def _embedding_dims(path: Path) -> Optional[int]:
    """Prefer the sentence-transformers pooling dim; fall back to hidden_size."""
    pooling = path / "1_Pooling" / "config.json"
    if pooling.is_file():
        try:
            dim = json.loads(pooling.read_text()).get("word_embedding_dimension")
            if isinstance(dim, int):
                return dim
        except (json.JSONDecodeError, OSError):
            pass
    cfg = path / "config.json"
    if cfg.is_file():
        try:
            data = json.loads(cfg.read_text())
            for key in ("hidden_size", "d_model", "dim"):
                if isinstance(data.get(key), int):
                    return data[key]
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _looks_like(name: str, hints: tuple) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in hints)


def _scan() -> List[Path]:
    """Model directories under the search roots, at most two levels deep."""
    found: List[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if _is_model_dir(entry):
                found.append(entry)
                continue
            for nested in sorted(entry.glob("snapshots/*")):
                if nested.is_dir() and _is_model_dir(nested):
                    found.append(nested)
    return found


def _pick(candidates: List[Path], hints: tuple, preferred: str) -> Optional[Path]:
    if Path(preferred).is_dir() and _is_model_dir(Path(preferred)):
        return Path(preferred)
    matches = [p for p in candidates if _looks_like(p.name, hints)]
    return matches[0] if matches else None


def _gpus() -> List[Dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        out = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "free_mb": round(free / 1024 / 1024),
                    "total_mb": round(total / 1024 / 1024),
                }
            )
        return out
    except Exception:
        return []


def _ollama() -> Dict[str, Any]:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import httpx

        resp = httpx.get(f"{host}/api/tags", timeout=1.5)
        resp.raise_for_status()
        models = [m.get("name") for m in resp.json().get("models", []) if m.get("name")]
        return {"running": True, "host": host, "models": models[:20]}
    except Exception:
        return {"running": False, "host": host, "models": []}


def _tiktoken_ok() -> bool:
    try:
        import tiktoken

        tiktoken.encoding_for_model("gpt-4o-mini")
        return True
    except Exception:
        return False


def detect() -> Dict[str, Any]:
    candidates = _scan()
    compressor = _pick(candidates, _COMPRESSOR_HINTS, DEFAULT_COMPRESSOR_PATH)
    embedder = _pick(candidates, _EMBEDDER_HINTS, DEFAULT_EMBEDDER_PATH)

    gpus = _gpus()
    best = max(gpus, key=lambda g: g["free_mb"]) if gpus else None
    device = f"cuda:{best['index']}" if best else "cpu"

    dims = _embedding_dims(embedder) if embedder else None

    return {
        "compressor": {"path": str(compressor) if compressor else None, "found": compressor is not None},
        "embedder": {
            "path": str(embedder) if embedder else None,
            "found": embedder is not None,
            "dims": dims,
        },
        "device": device,
        "gpus": gpus,
        "ollama": _ollama(),
        "tiktoken_ok": _tiktoken_ok(),
        "searched": [str(r) for r in SEARCH_ROOTS],
        "candidates": [str(p) for p in candidates][:30],
    }
