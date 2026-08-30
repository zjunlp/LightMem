"""Configuration schema, presets, form fields, and validation."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional

from .settings import (
    DEFAULT_COMPRESSOR_PATH,
    DEFAULT_EMBEDDER_PATH,
    QDRANT_DIR,
    LOG_DIR,
)


def json_schema() -> Dict[str, Any]:
    from lightmem.configs.base import BaseMemoryConfigs

    return BaseMemoryConfigs.model_json_schema()


def _field(path: str, label: str, label_zh: str, kind: str, **kw: Any) -> Dict[str, Any]:
    f = {"path": path, "label": label, "label_zh": label_zh, "kind": kind}
    f.update(kw)
    return f


def ui_spec() -> List[Dict[str, Any]]:
    """Return the fields exposed in the settings UI."""
    return [
        {
            "id": "pipeline",
            "title": "Pipeline",
            "title_zh": "流水线",
            "subtitle": "",
            "subtitle_zh": "",
            "fields": [
                _field("pre_compress", "Token compression", "token 压缩", "bool"),
                _field("topic_segment", "Topic segmentation", "主题切分", "bool"),
                _field("pre_compressor.configs.llmlingua_config.model_name",
                       "Compressor model path", "压缩器模型路径", "path",
                       showIf="pre_compress"),
                _field("pre_compressor.configs.llmlingua_config.device_map",
                       "Compressor device", "压缩器设备", "select",
                       options=["cuda:0", "cuda:1", "cpu"], showIf="pre_compress"),
            ],
        },
        {
            "id": "storage",
            "title": "Model & storage",
            "title_zh": "模型与存储",
            "subtitle": "",
            "subtitle_zh": "",
            "fields": [
                _field("memory_manager.model_name", "LLM backend", "LLM 后端", "select",
                       options=["openai", "deepseek", "ollama", "vllm", "transformers"],
                       help="Any OpenAI-compatible gateway; pick ollama to run locally.",
                       help_zh="本地跑选 ollama"),
                _field("memory_manager.configs.model", "Extraction model", "抽取模型", "model",
                       help="e.g. gpt-4o-mini, gemma3:latest",
                       help_zh="如 gpt-4o-mini、gemma3:latest"),
                _field("memory_manager.configs.host", "Ollama host", "Ollama 地址", "text",
                       showIf="memory_manager.model_name==ollama"),
                _field("text_embedder.configs.model", "Embedding model path", "嵌入模型路径", "path"),
                _field("text_embedder.configs.model_kwargs.device", "Embedding device", "嵌入设备", "select",
                       options=["cuda:0", "cuda:1", "cpu"]),
                _field("embedding_retriever.configs.collection_name", "Collection", "集合名", "text",
                       help="One collection = one memory store",
                       help_zh="一个集合 = 一个独立记忆库"),
            ],
        },
        {
            "id": "strategy",
            "title": "Indexing & retrieval strategy",
            "title_zh": "索引与检索策略",
            "subtitle": "",
            "subtitle_zh": "",
            "fields": [
                _field("index_strategy", "Index strategy", "索引策略", "select",
                       options=["embedding", "context", "hybrid"]),
                _field("retrieve_strategy", "Retrieve strategy", "检索策略", "select",
                       options=["embedding", "context", "hybrid"],
                       help="Must match the index strategy",
                       help_zh="需与索引策略一致"),
                _field("update", "Update mode", "更新模式", "select", options=["offline", "online"],
                       help="online is a no-op in this release",
                       help_zh="online 在当前版本是空实现"),
                _field("extract_threshold", "Extraction threshold", "抽取阈值", "number",
                       step=0.05, min=0, max=1,
                       help="Lower extracts more eagerly",
                       help_zh="越低越积极"),
            ],
        },
    ]


def _base_preset(collection: str) -> Dict[str, Any]:
    return {
        "memory_manager": {
            "model_name": "openai",
            "configs": {"model": "gpt-4o-mini", "max_tokens": 16000, "temperature": 0.1},
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "extraction_mode": "flat",
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": DEFAULT_EMBEDDER_PATH,
                "embedding_dims": 384,
                "model_kwargs": {"device": "cuda:0"},
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection,
                "embedding_model_dims": 384,
                "path": str(QDRANT_DIR / collection),
                "on_disk": True,
            },
        },
        "update": "offline",
        "logging": {
            "level": "DEBUG",
            "console_enabled": True,
            "file_enabled": True,
            "log_dir": str(LOG_DIR),
        },
    }


def presets() -> List[Dict[str, Any]]:
    full = _base_preset("lightmem_full")
    full.update(
        {
            "pre_compress": True,
            "pre_compressor": {
                "model_name": "llmlingua-2",
                "configs": {
                    "llmlingua_config": {
                        "model_name": DEFAULT_COMPRESSOR_PATH,
                        "device_map": "cuda:0",
                        "use_llmlingua2": True,
                    }
                },
            },
            "topic_segment": True,
            "precomp_topic_shared": True,
            "topic_segmenter": {"model_name": "llmlingua-2"},
            "extract_threshold": 0.1,
        }
    )

    light = _base_preset("lightmem_light")
    light.update({"pre_compress": False, "topic_segment": False, "extract_threshold": 0.5})

    segment_only = _base_preset("lightmem_segment")
    segment_only.update(
        {
            "pre_compress": False,
            "topic_segment": True,
            "precomp_topic_shared": False,
            "topic_segmenter": {"model_name": "llmlingua-2"},
            "extract_threshold": 0.1,
        }
    )

    return [
        {
            "id": "full",
            "name": "Full pipeline",
            "name_zh": "完整流水线",
            "description": "Best quality. Uses both local models.",
            "description_zh": "效果最好，用上两个本地模型",
            "config": full,
        },
        {
            "id": "segment",
            "name": "Segmentation only",
            "name_zh": "仅主题切分",
            "description": "Segmentation only. One model instead of two.",
            "description_zh": "只做主题切分，省一个模型",
            "config": segment_only,
        },
        {
            "id": "light",
            "name": "Minimal (no local models)",
            "name_zh": "最小配置（不用本地模型）",
            "description": "No local models. Quickest to get running.",
            "description_zh": "不用本地模型，最快跑起来",
            "config": light,
        },
    ]


def _get(config: Dict[str, Any], path: str) -> Any:
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def validate(config: Dict[str, Any]) -> Dict[str, Any]:
    """Type-check via pydantic and add the practical warnings the schema misses."""
    from pydantic import ValidationError

    from lightmem.configs.base import BaseMemoryConfigs

    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    probe = copy.deepcopy(config)
    try:
        BaseMemoryConfigs(**probe)
    except ValidationError as exc:
        for err in exc.errors():
            errors.append(
                {"path": ".".join(str(p) for p in err.get("loc", ())), "message": err.get("msg", "")}
            )
    except Exception as exc:
        errors.append({"path": "", "message": f"{type(exc).__name__}: {exc}"})

    retriever_cfg = _get(config, "embedding_retriever.configs") or {}
    path = retriever_cfg.get("path")
    if path and not retriever_cfg.get("on_disk"):
        exists = os.path.isdir(path)
        warnings.append(
            {
                "level": "danger",
                "path": "embedding_retriever.configs.on_disk",
                "message": (
                    "on_disk is off, so LightMem deletes the entire vector database "
                    f"directory ({path}) every time the instance is built."
                    + (" That directory exists now and will be erased." if exists else "")
                ),
                "message_zh": (
                    f"on_disk 已关闭，每次构建实例时 LightMem 都会删除整个向量库目录（{path}）。"
                    + ("该目录当前存在，将被清空。" if exists else "")
                ),
            }
        )

    emb_dims = _get(config, "text_embedder.configs.embedding_dims")
    store_dims = retriever_cfg.get("embedding_model_dims")
    if emb_dims and store_dims and emb_dims != store_dims:
        errors.append(
            {
                "path": "embedding_retriever.configs.embedding_model_dims",
                "message": f"Vector store expects {store_dims} dims but the embedder produces {emb_dims}.",
                "message_zh": f"向量库期望 {store_dims} 维，但嵌入模型输出 {emb_dims} 维。",
            }
        )

    for path_key, label, label_zh in (
        ("pre_compressor.configs.llmlingua_config.model_name", "Compressor model", "压缩器模型"),
        ("text_embedder.configs.model", "Embedding model", "嵌入模型"),
    ):
        value = _get(config, path_key)
        needs_check = path_key.startswith("pre_compressor") and config.get("pre_compress")
        needs_check = needs_check or path_key.startswith("text_embedder")
        if needs_check and isinstance(value, str) and value.startswith("/") and not os.path.isdir(value):
            errors.append(
                {
                    "path": path_key,
                    "message": f"{label} directory not found: {value}",
                    "message_zh": f"{label_zh}目录不存在：{value}",
                }
            )

    if config.get("topic_segment") and config.get("precomp_topic_shared") and not config.get("pre_compress"):
        errors.append(
            {
                "path": "precomp_topic_shared",
                "message": "Sharing weights with the compressor requires pre_compress to be enabled.",
                "message_zh": "与压缩器共享权重需要先启用 pre_compress。",
            }
        )

    if _get(config, "text_embedder.configs.model_kwargs") is None and \
            _get(config, "text_embedder.model_name") == "huggingface":
        warnings.append(
            {
                "level": "warn",
                "path": "text_embedder.configs.model_kwargs",
                "message": "model_kwargs is empty; it will be sent as {} to avoid a TypeError in LightMem.",
                "message_zh": "model_kwargs 为空，将以 {} 发送，以避免 LightMem 内部抛 TypeError。",
            }
        )

    if config.get("index_strategy") in ("embedding", "hybrid") and not config.get("text_embedder"):
        errors.append(
            {
                "path": "text_embedder",
                "message": "An embedder is required for embedding indexing.",
                "message_zh": "使用 embedding 索引需要配置嵌入模型。",
            }
        )

    if config.get("retrieve_strategy") in ("embedding", "hybrid") and not config.get("embedding_retriever"):
        errors.append(
            {
                "path": "embedding_retriever",
                "message": "A vector store is required for embedding retrieval.",
                "message_zh": "使用 embedding 检索需要配置向量库。",
            }
        )

    if config.get("update") == "online":
        warnings.append(
            {
                "level": "warn",
                "path": "update",
                "message": "online_update() is a no-op in this LightMem release -- nothing will be written.",
                "message_zh": "该版本 LightMem 的 online_update() 是空实现，不会写入任何内容。",
            }
        )

    manager_backend = _get(config, "memory_manager.model_name")
    if manager_backend in ("openai", "deepseek"):
        from . import secrets_store

        has_inline = bool(_get(config, "memory_manager.configs.api_key"))
        if not has_inline and not secrets_store.load()["api_key"]:
            errors.append(
                {
                    "path": "memory_manager.configs.api_key",
                    "message": (
                        f"The {manager_backend} manager builds its client during construction, so an "
                        "API key is required before the instance can be created at all."
                    ),
                    "message_zh": (
                        f"{manager_backend} manager 在构造时就会建客户端，因此必须先填 API Key 才能启动实例。"
                    ),
                }
            )

    try:
        import tiktoken

        tiktoken.encoding_for_model("gpt-4o-mini")
    except Exception as exc:
        errors.append(
            {
                "path": "tiktoken",
                "message": (
                    "The tiktoken encoding cache is not primed, so LightMem would hang on the TLS "
                    f"handshake to openaipublic instead of failing. Run web/scripts/prime_tiktoken_cache.sh. ({exc})"
                ),
                "message_zh": (
                    "tiktoken 编码缓存未预热，LightMem 会卡在到 openaipublic 的 TLS 握手上而不是报错。"
                    "执行一次 web/scripts/prime_tiktoken_cache.sh 即可。"
                ),
            }
        )

    return {"ok": not errors, "errors": errors, "warnings": warnings}
