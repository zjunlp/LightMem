from __future__ import annotations
import os
import json
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator

from .base import BaseMemoryLayer
from token_monitor import get_tokenizer_for_model
from litellm import token_counter as litellm_token_counter

class FullContextConfig(BaseModel):
    """Configuration: Full-Text Memory layer (simple full-context storage and retrieval).

    This layer stores full message text with minimal dependencies and provides:
    - Append single/multiple messages
    - Simple retrieval (based on substring/keyword token matching)
    - Delete and update
    - Save and load
    """

    user_id: str = Field(..., description="Unique user ID")
    save_dir: str = Field(
        default="full_context",
        description="Directory to save memory (contains config.json and serialized .pkl data)",
    )

    context_window: int = Field(
        default=1000000,
        description="Context window size (approximate by token count); exceeding drops earliest messages",
        gt=0,
    )

    # Not used; kept for compatibility with common scripts
    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="LLM backend provider (kept for consistency).",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name (kept for consistency).",
    )

    @model_validator(mode="after")
    def _validate_save_dir(self) -> FullContextConfig:
        if os.path.isfile(self.save_dir):
            raise AssertionError(
                f"Provided path ({self.save_dir}) should be a directory, not a file"
            )
        return self


class FullContextLayer(BaseMemoryLayer):
    """Full-Text Memory layer: store complete messages and provide basic retrieval."""

    layer_type: str = "full_context"

    def __init__(self, config: FullContextConfig) -> None:
        self.config = config
        # Memory structure: {id: {"content": str, "role": str, "timestamp": float, "metadata": {...}}}
        self._memories: Dict[str, Dict[str, Any]] = {}
        self._ordered_ids: List[str] = []

        # Token-related: track token count per message and total tokens
        self._token_per_id: Dict[str, int] = {}
        self._total_tokens: int = 0

        # Initialize tokenizer (shares logic with token_monitor)
        self._tokenizer = get_tokenizer_for_model(self.config.llm_model)

    # Basic utilities
    def _gen_id(self) -> str:
        # Use timestamp + count to ensure basic uniqueness
        nid = f"{int(time.time()*1000)}-{len(self._ordered_ids)}"
        return nid

    def _count_tokens(self, text: str) -> int:
        """Count tokens for the given text.

        Reuse litellm's token_counter and the same custom_tokenizer returned by
        get_tokenizer_for_model (shared with token_monitor) to ensure consistent counting.
        """
        try:
            return litellm_token_counter(
                model=self.config.llm_model,
                custom_tokenizer=self._tokenizer,
                text=text,
            )
        except Exception:
            # Fallback: if external dependency is unavailable, approximate by character count
            return len(text)

    def _recalculate_tokens(self) -> None:
        """Recompute token count per message and total tokens based on current memory.

        Used after loading from older versions or when overall calibration is needed.
        """
        self._token_per_id.clear()
        self._total_tokens = 0
        for mid in self._ordered_ids:
            mem = self._memories.get(mid)
            if not mem:
                continue
            content = mem.get("content", "") or ""
            tks = self._count_tokens(content)
            self._token_per_id[mid] = tks
            self._total_tokens += tks

    def _ensure_capacity(self) -> None:
        """Ensure total tokens do not exceed the context_window after writes.

        If exceeding the limit, remove messages from the oldest until constraints are met.
        """
        max_tokens = getattr(self.config, "context_window", None)
        if not max_tokens or max_tokens <= 0:
            return

        while self._ordered_ids and self._total_tokens > max_tokens:
            oldest_id = self._ordered_ids.pop(0)
            # Subtract from token statistics
            tks = self._token_per_id.pop(oldest_id, 0)
            self._total_tokens -= tks
            # Remove actual content
            self._memories.pop(oldest_id, None)

    # Interface implementations
    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        if "role" not in message or "content" not in message:
            raise KeyError("message must contain 'role' and 'content'")
        ts = kwargs.get("timestamp", time.time())
        mid = self._gen_id()
        payload = {
            "id": mid,
            "role": message["role"],
            "content": message["content"],
            "name": message.get("name"),
            "timestamp": ts,
            "metadata": {k: v for k, v in kwargs.items() if k != "timestamp"},
        }
        self._memories[mid] = payload
        self._ordered_ids.append(mid)
        self._ensure_capacity()

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        for m in messages:
            self.add_message(m, **kwargs)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        **kwargs,
    ) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
        """
        Parameters:
        - query: search keyword (if empty string and mode not explicitly specified, fallback to full)
        - k: override the default number of returned items; when k = -1, return all.
        """

        results: List[Dict[str, Union[str, Dict[str, Any]]]] = []

        count = 0
        # Scan in reverse order to prioritize latest content
        for mid in reversed(self._ordered_ids):
            mem = self._memories[mid]
            used_content = {
                "Time": mem["timestamp"],
                "Memory": mem["content"], 
            }
            results.append(
                {
                    "content": mem["content"],
                    "metadata": {
                        key: value
                        for key, value in mem.items() if key != "content"
                    },
                    "used_content": '\n'.join(
                        [f"{key}: {value}" for key, value in used_content.items()]
                    )
                }
            )
            count += 1
            if k != -1 and count >= k:
                break
        return results

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            # Update token statistics first
            tks = self._token_per_id.pop(memory_id, 0)
            self._total_tokens -= tks

            self._memories.pop(memory_id)
            try:
                self._ordered_ids.remove(memory_id)
            except ValueError:
                pass
            return True
        return False

    def update(self, memory_id: str, **kwargs) -> bool:
        mem = self._memories.get(memory_id)
        if not mem:
            return False

        # Updating content requires synchronizing token statistics
        if "content" in kwargs:
            new_content = kwargs["content"]
            old_tokens = self._token_per_id.get(memory_id, 0)
            new_tokens = self._count_tokens(new_content)

            mem["content"] = new_content

            self._token_per_id[memory_id] = new_tokens
            self._total_tokens += new_tokens - old_tokens

            # Ensure capacity after the update
            self._ensure_capacity()

        if "role" in kwargs:
            mem["role"] = kwargs["role"]
        if "timestamp" in kwargs:
            mem["timestamp"] = kwargs["timestamp"]

        other = {
            k: v
            for k, v in kwargs.items()
            if k not in {"content", "role", "timestamp"}
        }
        if other:
            meta = mem.get("metadata", {})
            meta.update(other)
            mem["metadata"] = meta
        return True

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json
        cfg_path = os.path.join(self.config.save_dir, "config.json")
        cfg = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "context_window": self.config.context_window,
        }
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)

        # Write memory snapshot .pkl
        payload = {
            "ordered_ids": self._ordered_ids,
            "memories": self._memories,
        }
        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        uid = user_id or self.config.user_id
        cfg_path = os.path.join(self.config.save_dir, "config.json")
        pkl_path = os.path.join(self.config.save_dir, f"{uid}.pkl")
        if not (os.path.exists(cfg_path) and os.path.exists(pkl_path)):
            return False

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        if uid != cfg_dict.get("user_id"):
            raise ValueError(
                f"Config user_id ({cfg_dict.get('user_id')}) does not match requested ({uid})"
            )

        # Update configuration (excluding save_dir)
        self.config = FullContextConfig(
            user_id=cfg_dict["user_id"],
            save_dir=self.config.save_dir,
            context_window=cfg_dict.get("context_window", 8196),
        )

        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self._ordered_ids = payload.get("ordered_ids", [])
        self._memories = payload.get("memories", {})
        
        # Recalculate token statistics and enforce capacity
        self._recalculate_tokens()
        self._ensure_capacity()
        return True
