"""Local credential storage with masked browser responses."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from .settings import SECRETS_FILE, ensure_dirs

_DEFAULT: Dict[str, Any] = {
    "api_key": "",
    "base_url": "",
    "model": "gpt-4o-mini",
    "answer_model": "",
}


def load() -> Dict[str, Any]:
    ensure_dirs()
    if not SECRETS_FILE.exists():
        return dict(_DEFAULT)
    try:
        data = json.loads(SECRETS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    merged = dict(_DEFAULT)
    merged.update({k: v for k, v in data.items() if k in _DEFAULT})
    return merged


class InvalidSecret(ValueError):
    pass


def save(values: Dict[str, Any]) -> Dict[str, Any]:
    current = load()
    for key in _DEFAULT:
        if key in values and values[key] is not None:
            value = values[key]
            if isinstance(value, str):
                value = value.strip()
            if key == "api_key" and value.lower().startswith(("http://", "https://")):
                raise InvalidSecret(
                    "That looks like a URL, not an API key. The address belongs in Base URL."
                )
            current[key] = value
    ensure_dirs()
    SECRETS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    try:
        os.chmod(SECRETS_FILE, 0o600)
    except OSError:
        pass
    return current


def mask(key: str) -> str:
    """Return a non-usable preview of a key."""
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


def public_view() -> Dict[str, Any]:
    s = load()
    return {
        "api_key_masked": mask(s["api_key"]),
        "has_api_key": bool(s["api_key"]),
        "base_url": s["base_url"],
        "model": s["model"],
        "answer_model": s["answer_model"],
    }


def inject_into_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing credentials from the local secret store."""
    s = load()
    manager = config.get("memory_manager")
    if isinstance(manager, dict):
        cfgs = manager.setdefault("configs", {})
        if isinstance(cfgs, dict):
            if not cfgs.get("api_key") and s["api_key"]:
                cfgs["api_key"] = s["api_key"]
            model_name = manager.get("model_name")
            url_field = {
                "openai": "openai_base_url",
                "deepseek": "deepseek_base_url",
                "vllm": "vllm_base_url",
            }.get(model_name)
            if url_field and not cfgs.get(url_field) and s["base_url"]:
                cfgs[url_field] = s["base_url"]

    embedder = config.get("text_embedder")
    if isinstance(embedder, dict) and embedder.get("model_name") == "openai":
        cfgs = embedder.setdefault("configs", {})
        if isinstance(cfgs, dict):
            if not cfgs.get("api_key") and s["api_key"]:
                cfgs["api_key"] = s["api_key"]
            if not cfgs.get("openai_base_url") and s["base_url"]:
                cfgs["openai_base_url"] = s["base_url"]
    return config


def redact_config(config: Any) -> Any:
    """Deep-copy a config with any api_key masked, for sending to the browser."""
    if isinstance(config, dict):
        out = {}
        for k, v in config.items():
            if k == "api_key" and isinstance(v, str):
                out[k] = mask(v)
            else:
                out[k] = redact_config(v)
        return out
    if isinstance(config, list):
        return [redact_config(v) for v in config]
    return config
