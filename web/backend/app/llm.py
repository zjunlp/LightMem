"""OpenAI-compatible client helpers: connectivity checks and the answer step."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import secrets_store


def _client(api_key: Optional[str] = None, base_url: Optional[str] = None):
    from openai import OpenAI

    stored = secrets_store.load()
    key = api_key or stored["api_key"]
    url = base_url or stored["base_url"]
    if not key:
        raise ValueError("No API key configured. Enter one on the Settings page first.")
    kwargs: Dict[str, Any] = {"api_key": key}
    if url:
        kwargs["base_url"] = url
    return OpenAI(**kwargs)


def ping(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify credentials by listing models, then round-trip one tiny completion."""
    stored = secrets_store.load()
    model = model or stored["model"] or "gpt-4o-mini"
    out: Dict[str, Any] = {"ok": False, "model": model}

    try:
        client = _client(api_key, base_url)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    t0 = time.time()
    try:
        listing = client.models.list()
        names = [m.id for m in listing.data][:200]
        out["models_visible"] = len(names)
        out["models_sample"] = sorted(names)[:25]
        out["model_listed"] = model in names
    except Exception as exc:
        out["models_error"] = f"{type(exc).__name__}: {exc}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=8,
            temperature=0,
        )
        out["ok"] = True
        out["reply"] = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        if usage:
            out["usage"] = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"

    out["latency_ms"] = round((time.time() - t0) * 1000)
    return out


def list_models(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    backend: Optional[str] = None,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    """List models reported by the configured endpoint."""
    if backend == "ollama":
        try:
            import httpx

            url = (host or "http://localhost:11434").rstrip("/")
            resp = httpx.get(f"{url}/api/tags", timeout=4)
            resp.raise_for_status()
            names = [m["name"] for m in resp.json().get("models", []) if m.get("name")]
            return {"models": sorted(names), "error": None}
        except Exception as exc:
            return {"models": [], "error": f"{type(exc).__name__}: {exc}"}

    try:
        client = _client(api_key, base_url)
        names = [m.id for m in client.models.list().data]
        return {"models": sorted(set(names)), "error": None}
    except Exception as exc:
        return {"models": [], "error": f"{type(exc).__name__}: {exc}"}


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's long-term memory. "
    "Use the retrieved memories below to answer. Each memory is prefixed with its "
    "timestamp and weekday. If the memories do not contain the answer, say so plainly "
    "instead of guessing."
)


def build_context(memories: List[Dict[str, Any]]) -> str:
    lines = []
    for i, m in enumerate(memories, 1):
        stamp = " ".join(x for x in (m.get("time_stamp", ""), m.get("weekday", "")) if x)
        lines.append(f"[{i}] {stamp} {m.get('memory', '')}".strip())
    return "\n".join(lines)


def build_messages(
    question: str,
    memories: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    context = build_context(memories)
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    user = (
        f"Retrieved memories:\n{context}\n\nQuestion: {question}"
        if context
        else f"No memories were retrieved.\n\nQuestion: {question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def answer(
    question: str,
    memories: List[Dict[str, Any]],
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    stored = secrets_store.load()
    model = model or stored["answer_model"] or stored["model"] or "gpt-4o-mini"
    messages = build_messages(question, memories, system_prompt)

    client = _client()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    usage = getattr(resp, "usage", None)
    return {
        "answer": resp.choices[0].message.content or "",
        "model": model,
        "messages": messages,
        "latency_ms": round((time.time() - t0) * 1000),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        if usage
        else None,
    }
