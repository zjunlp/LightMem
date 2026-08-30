"""Lifecycle and operations for the active LightMemory instance."""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import secrets_store
from .jobs import Job, manager as job_manager

log = logging.getLogger("web.instance")

_LOCK = threading.RLock()

_instance: Any = None
_config: Optional[Dict[str, Any]] = None
_created_at: Optional[float] = None
_init_seconds: Optional[float] = None
_last_error: Optional[str] = None


class InstanceNotReady(RuntimeError):
    pass


def _close_qdrant(obj: Any) -> None:
    client = getattr(obj, "client", None)
    if client is None:
        return
    try:
        client.close()
    except Exception as exc:  # pragma: no cover - best effort teardown
        log.warning("Failed to close Qdrant client cleanly: %s", exc)


def dispose() -> None:
    """Tear the current instance down and release the Qdrant directory lock."""
    global _instance, _config, _created_at, _init_seconds
    with _LOCK:
        inst = _instance
        if inst is not None:
            for attr in ("embedding_retriever", "summary_retriever"):
                retriever = getattr(inst, attr, None)
                if retriever is not None:
                    _close_qdrant(retriever)
        _instance = None
        _config = None
        _created_at = None
        _init_seconds = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _reset_module_globals() -> None:
    """Topic ids and the summary time pointer are module globals in LightMem.

    Left alone they keep counting across rebuilds, so a fresh instance would
    start numbering topics at wherever the previous run stopped.
    """
    import lightmem.memory.lightmem as lm

    lm.GLOBAL_TOPIC_IDX = 0
    lm.GLOBAL_LAST_SUMMARY_TIME = None


def initialize(raw_config: Dict[str, Any], job: Optional[Job] = None) -> Dict[str, Any]:
    """Build a new LightMemory from a config dict, replacing any existing one."""
    global _instance, _config, _created_at, _init_seconds, _last_error

    from lightmem.memory.lightmem import LightMemory

    config = secrets_store.inject_into_config(dict(raw_config))
    _normalize_config(config)

    with _LOCK:
        if job:
            job_manager.stage(job, "dispose", {"message": "Releasing previous instance"})
        dispose()
        _reset_module_globals()

        if job:
            job_manager.stage(job, "construct", {"message": "Loading components"})
        started = time.time()
        try:
            instance = LightMemory.from_config(config)
        except Exception as exc:
            _last_error = f"{type(exc).__name__}: {exc}"
            raise
        elapsed = time.time() - started

        # LoggingConfig.apply() runs inside the constructor and wipes root
        # handlers, so the log bridge has to be put back.
        job_manager.install_log_bridge()

        _instance = instance
        _config = config
        _created_at = time.time()
        _init_seconds = elapsed
        _last_error = None

    if job:
        job_manager.stage(job, "ready", {"seconds": round(elapsed, 2)})
    return status()


def _normalize_config(config: Dict[str, Any]) -> None:
    """Patch up shapes that would otherwise blow up deep inside LightMem."""
    embedder = config.get("text_embedder")
    if isinstance(embedder, dict):
        cfgs = embedder.get("configs")
        if isinstance(cfgs, dict) and cfgs.get("model_kwargs") is None:
            # TextEmbedderHuggingface does SentenceTransformer(model,
            # **config.model_kwargs) -- None raises TypeError.
            cfgs["model_kwargs"] = {}


def get() -> Any:
    with _LOCK:
        if _instance is None:
            raise InstanceNotReady(
                "LightMem instance is not running. Start it on the Settings page first."
            )
        return _instance


def peek() -> Any:
    return _instance


def status() -> Dict[str, Any]:
    with _LOCK:
        inst = _instance
        if inst is None:
            return {
                "ready": False,
                "config": None,
                "components": [],
                "created_at": None,
                "init_seconds": None,
                "last_error": _last_error,
            }

        components: List[Dict[str, Any]] = []

        def add(name: str, attr: str, note: str = "") -> None:
            obj = getattr(inst, attr, None)
            if obj is not None:
                components.append(
                    {"name": name, "impl": type(obj).__name__, "note": note}
                )

        add("Pre-compressor", "compressor", "sensory-stage token compression")
        add("Topic segmenter", "segmenter", "sensory-stage topic grouping")
        add("Memory manager (LLM)", "manager", "metadata + summary extraction")
        add("Text embedder", "text_embedder", "")
        add("Embedding retriever", "embedding_retriever", "")
        add("Context retriever", "context_retriever", "")
        add("Summary retriever", "summary_retriever", "")
        add("Graph memory", "graph", "")

        collection = None
        points = None
        retriever = getattr(inst, "embedding_retriever", None)
        if retriever is not None:
            collection = getattr(retriever, "collection_name", None)
            try:
                points = retriever.client.count(collection_name=collection).count
            except Exception:
                points = None

        return {
            "ready": True,
            "config": secrets_store.redact_config(_config),
            "components": components,
            "created_at": _created_at,
            "init_seconds": _init_seconds,
            "collection": collection,
            "points": points,
            "retrieve_strategy": getattr(inst, "retrieve_strategy", None),
            "last_error": None,
            "gpu": _gpu_snapshot(),
        }


def _gpu_snapshot() -> List[Dict[str, Any]]:
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
                    "total_mb": round(total / 1024 / 1024),
                    "used_mb": round((total - free) / 1024 / 1024),
                }
            )
        return out
    except Exception:
        return []


def add_memory(
    messages: List[Dict[str, Any]],
    force_segment: bool = False,
    force_extract: bool = False,
    metadata_prompt: Optional[Any] = None,
) -> Dict[str, Any]:
    inst = get()
    with _LOCK:
        before = _point_count(inst)
        try:
            result = inst.add_memory(
                messages,
                metadata_prompt,
                force_segment=force_segment,
                force_extract=force_extract,
            )
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'get'" in str(exc):
                raise RuntimeError(
                    "Every extraction call to the LLM failed, so no usage was returned. "
                    "The provider's error is on the line above, tagged `stdout` — usually a "
                    "bad API key, an unreachable base URL, or a model name the gateway "
                    "does not serve. Check the key and base URL on the Settings page."
                ) from exc
            raise
        after = _point_count(inst)
    payload = dict(result or {})
    payload["points_before"] = before
    payload["points_after"] = after
    payload["points_added"] = (
        after - before if before is not None and after is not None else None
    )
    return payload


def offline_update(
    top_k: int = 20, keep_top_n: int = 10, score_threshold: float = 0.8
) -> Dict[str, Any]:
    inst = get()
    with _LOCK:
        before = _point_count(inst)
        t0 = time.time()
        inst.construct_update_queue_all_entries(top_k=top_k, keep_top_n=keep_top_n)
        t1 = time.time()
        inst.offline_update_all_entries(score_threshold=score_threshold)
        t2 = time.time()
        after = _point_count(inst)
        consolidated = _count_consolidated(inst)
    return {
        "queue_seconds": round(t1 - t0, 2),
        "update_seconds": round(t2 - t1, 2),
        "points_before": before,
        "points_after": after,
        "consolidated": consolidated,
    }


def retrieve(
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search with the configured retriever and return full payloads."""
    inst = get()
    with _LOCK:
        embedder = getattr(inst, "text_embedder", None)
        retriever = getattr(inst, "embedding_retriever", None)
        if embedder is None or retriever is None:
            raise InstanceNotReady(
                "Embedding retrieval is unavailable. Set index_strategy and "
                "retrieve_strategy to 'embedding' or 'hybrid'."
            )
        vector = embedder.embed(query)
        hits = retriever.search(
            query_vector=vector, limit=limit, filters=filters or None, return_full=True
        )
    results = []
    for hit in hits:
        payload = hit.get("payload") or {}
        results.append(
            {
                "id": hit.get("id"),
                "score": hit.get("score"),
                "time_stamp": payload.get("time_stamp", ""),
                "weekday": payload.get("weekday", ""),
                "memory": payload.get("memory", ""),
                "original_memory": payload.get("original_memory", ""),
                "compressed_memory": payload.get("compressed_memory", ""),
                "topic_id": payload.get("topic_id"),
                "topic_summary": payload.get("topic_summary", ""),
                "category": payload.get("category", ""),
                "subcategory": payload.get("subcategory", ""),
                "memory_class": payload.get("memory_class", ""),
                "speaker_id": payload.get("speaker_id", ""),
                "speaker_name": payload.get("speaker_name", ""),
                "consolidated": payload.get("consolidated", False),
                "bam_tags": payload.get("bam_tags", []),
            }
        )
    return results


def browse(
    limit: int = 50,
    offset: Any = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    inst = get()
    with _LOCK:
        retriever = getattr(inst, "embedding_retriever", None)
        if retriever is None:
            raise InstanceNotReady("No embedding retriever configured.")
        points, next_offset = retriever.scroll(
            scroll_filter=filters or None,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    rows = []
    for p in points:
        payload = dict(getattr(p, "payload", None) or {})
        payload["id"] = getattr(p, "id", None)
        rows.append(payload)
    return rows, next_offset


def token_statistics() -> Dict[str, Any]:
    inst = get()
    with _LOCK:
        return inst.get_token_statistics()


def _point_count(inst: Any) -> Optional[int]:
    retriever = getattr(inst, "embedding_retriever", None)
    if retriever is None:
        return None
    try:
        return retriever.client.count(collection_name=retriever.collection_name).count
    except Exception:
        return None


def _count_consolidated(inst: Any) -> Optional[int]:
    retriever = getattr(inst, "embedding_retriever", None)
    if retriever is None:
        return None
    try:
        total = 0
        offset = None
        while True:
            points, offset = retriever.scroll(
                limit=200, offset=offset, with_payload=True, with_vectors=False
            )
            for p in points:
                if (getattr(p, "payload", None) or {}).get("consolidated"):
                    total += 1
            if offset is None:
                break
        return total
    except Exception:
        return None
