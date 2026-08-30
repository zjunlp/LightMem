"""HTTP API for the LightMem web console."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import autodetect, configspec, instance, llm, secrets_store
from .jobs import manager as job_manager

router = APIRouter(prefix="/api")


def _guard(fn, *args, **kwargs):
    """Turn instance/state errors into clean 4xx responses."""
    try:
        return fn(*args, **kwargs)
    except instance.InstanceNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "instance_ready": instance.peek() is not None,
        "busy": job_manager.busy,
        "queue_depth": job_manager.queue_depth(),
    }


class SecretsRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    answer_model: Optional[str] = None


@router.get("/secrets")
def get_secrets() -> Dict[str, Any]:
    return secrets_store.public_view()


@router.post("/secrets")
def set_secrets(req: SecretsRequest) -> Dict[str, Any]:
    values = {k: v for k, v in req.model_dump().items() if v is not None}
    try:
        secrets_store.save(values)
    except secrets_store.InvalidSecret as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return secrets_store.public_view()


@router.post("/secrets/ping")
def ping_llm(req: SecretsRequest) -> Dict[str, Any]:
    return llm.ping(api_key=req.api_key or None, base_url=req.base_url or None, model=req.model or None)


class ModelsRequest(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    backend: Optional[str] = None
    host: Optional[str] = None


@router.post("/models")
def list_models(req: ModelsRequest) -> Dict[str, Any]:
    return llm.list_models(
        api_key=req.api_key or None,
        base_url=req.base_url or None,
        backend=req.backend or None,
        host=req.host or None,
    )


@router.get("/autodetect")
def autodetect_env() -> Dict[str, Any]:
    return autodetect.detect()


@router.get("/config/schema")
def config_schema() -> Dict[str, Any]:
    return {
        "schema": configspec.json_schema(),
        "ui": configspec.ui_spec(),
        "presets": configspec.presets(),
    }


class ConfigRequest(BaseModel):
    config: Dict[str, Any]


@router.post("/config/validate")
def config_validate(req: ConfigRequest) -> Dict[str, Any]:
    return configspec.validate(req.config)


@router.get("/instance")
def instance_status() -> Dict[str, Any]:
    return instance.status()


@router.post("/instance/init")
def instance_init(req: ConfigRequest) -> Dict[str, Any]:
    check = configspec.validate(req.config)
    if not check["ok"]:
        raise HTTPException(status_code=400, detail={"message": "Invalid configuration", **check})

    config = req.config

    def run(job):
        return instance.initialize(config, job=job)

    job = job_manager.submit("instance.init", run, {"collection": _collection_of(config)})
    return {"job_id": job.id, "warnings": check["warnings"]}


def _collection_of(config: Dict[str, Any]) -> Optional[str]:
    try:
        return config["embedding_retriever"]["configs"]["collection_name"]
    except (KeyError, TypeError):
        return None


@router.post("/instance/dispose")
def instance_dispose() -> Dict[str, Any]:
    instance.dispose()
    return instance.status()


class Message(BaseModel):
    role: str
    content: str
    time_stamp: Optional[str] = None


class AddMemoryRequest(BaseModel):
    messages: List[Message]
    force_segment: bool = False
    force_extract: bool = False
    metadata_prompt: Optional[Any] = None


@router.post("/memory/add")
def memory_add(req: AddMemoryRequest) -> Dict[str, Any]:
    if instance.peek() is None:
        raise HTTPException(status_code=409, detail="Initialize the LightMem instance first.")
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    payload = [m.model_dump(exclude_none=True) for m in req.messages]

    def run(job):
        job_manager.stage(job, "add_memory", {"messages": len(payload)})
        return instance.add_memory(
            payload,
            force_segment=req.force_segment,
            force_extract=req.force_extract,
            metadata_prompt=req.metadata_prompt,
        )

    job = job_manager.submit(
        "memory.add",
        run,
        {
            "messages": len(payload),
            "force_segment": req.force_segment,
            "force_extract": req.force_extract,
        },
    )
    return {"job_id": job.id}


class OfflineUpdateRequest(BaseModel):
    top_k: int = Field(20, ge=1, le=200)
    keep_top_n: int = Field(10, ge=1, le=100)
    score_threshold: float = Field(0.8, ge=0.0, le=1.0)


@router.post("/update/offline")
def update_offline(req: OfflineUpdateRequest) -> Dict[str, Any]:
    if instance.peek() is None:
        raise HTTPException(status_code=409, detail="Initialize the LightMem instance first.")

    def run(job):
        job_manager.stage(job, "update_queue", {"top_k": req.top_k, "keep_top_n": req.keep_top_n})
        return instance.offline_update(
            top_k=req.top_k, keep_top_n=req.keep_top_n, score_threshold=req.score_threshold
        )

    job = job_manager.submit("update.offline", run, req.model_dump())
    return {"job_id": job.id}


class RetrieveRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=200)
    filters: Optional[Dict[str, Any]] = None


@router.post("/retrieve")
def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty.")
    t0 = time.time()
    results = _guard(instance.retrieve, req.query, req.limit, req.filters)
    return {
        "query": req.query,
        "count": len(results),
        "latency_ms": round((time.time() - t0) * 1000),
        "results": results,
    }


@router.get("/memories")
def memories(
    limit: int = Query(50, ge=1, le=200),
    offset: Optional[str] = None,
) -> Dict[str, Any]:
    """Enumerate stored entries without ranking."""
    rows, next_offset = _guard(instance.browse, limit, offset, None)
    return {
        "count": len(rows),
        "next_offset": str(next_offset) if next_offset is not None else None,
        "rows": rows,
    }


class ChatRequest(BaseModel):
    question: str
    limit: int = Field(10, ge=0, le=100)
    filters: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    selected_ids: Optional[List[str]] = None
    use_memory: bool = True


@router.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    retrieved: List[Dict[str, Any]] = []
    if req.use_memory and req.limit > 0:
        retrieved = _guard(instance.retrieve, req.question, req.limit, req.filters)

    used = retrieved
    if req.selected_ids is not None:
        wanted = set(req.selected_ids)
        used = [r for r in retrieved if str(r.get("id")) in wanted]

    try:
        result = llm.answer(
            req.question,
            used,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")

    result["retrieved"] = retrieved
    result["used_count"] = len(used)
    return result
@router.get("/jobs")
def list_jobs(limit: int = Query(30, ge=1, le=200)) -> List[Dict[str, Any]]:
    return job_manager.list(limit)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return job.detail()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Dict[str, Any]:
    ok = job_manager.cancel(job_id)
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="Only queued jobs can be cancelled; a running LightMem call cannot be interrupted safely.",
        )
    return {"cancelled": job_id}


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, after: int = 0) -> StreamingResponse:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")

    queue = job_manager.subscribe(job_id)

    async def stream():
        try:
            # Replay what already happened so a late subscriber sees the whole run.
            replayed = 0
            for event in list(job.events):
                if event["seq"] > after:
                    replayed = event["seq"]
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if job.status in ("succeeded", "failed", "cancelled"):
                yield f"data: {json.dumps({'kind': 'done', 'status': job.status, 'seq': replayed}, ensure_ascii=False)}\n\n"
                return

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("kind") == "done":
                    return
        finally:
            job_manager.unsubscribe(job_id, queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
