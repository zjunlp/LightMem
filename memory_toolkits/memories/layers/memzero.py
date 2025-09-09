from __future__ import annotations
from .base import BaseMemoryLayer 
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from typing import (
    Literal, 
    List, 
    Dict, 
    Any,
    Optional, 
)
import os
from baselines.mem0 import Memory

class MemZeroConfig(BaseModel):
    """The default configuration for MemZero"""

    # Config for memory
    user_id: str = Field(..., description="The user id of the memory system.")

    save_dir: str = Field(
        default = "vector_store/memzero", 
        description = "The directory to save the memory."
    )

    collection_name: str = Field(
        default = "guest",
        description = "The name of the collection to save the memory.",
    )
    # Config for retriever
    retriever_name_or_path: str = Field(
        default="all-MiniLM-L6-v2",
        description="The name or path of the retriever model to use.",
    )

    embedding_model_dims: int = Field(
        default = 384,
        description = "The dimension of the embedding model.",
    )

    use_gpu: str = Field(
        default = "cpu",
        description = "The GPU to use for the embedding model.",
    )
    # Config for LLM
    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="The backend to use for the LLM. Currently, only openai and ollama are supported.",
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="The base backbone model to use.",
    )

    api_key: str | None = Field(
        default = None,
        description="The API key to use for the LLM. It is used for openai backend. "
        "If not provided, the API key will be loaded from the environment variable.",
    )

    base_url: str = Field(
            default = None,
            description = "The base URL of the LLM backend.",
        )
        
    temperature: float = Field(
        default = 0.1,
        description = "The temperature to use for the LLM.",
    )

    max_tokens: int = Field(
        default = 1024,
        description = "The maximum number of tokens to generate.",
    )

    @model_validator(mode="after")
    def _validate_save_dir(self) -> MemZeroConfig:
        if os.path.isfile(self.save_dir):
            raise AssertionError(f"Provided path ({self.save_dir}) should be a directory, not a file")
        return self 

class MemZeroLayer(BaseMemoryLayer):
    layer_type: str = "memzero"

    def __init__(self, config: MemZeroConfig) -> None:
        """Create an interface of MemZero. The implemenation is based on the 
        [official implementation](https://github.com/mem0ai/mem0)."""
        self.memory_config = {
            {
                "llm": {
                    "provider": config.llm_backend,
                    "config":{
                        "model": config.llm_model,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "api_key": config.api_key,
                        "base_url": config.base_url,
                    }
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": config.collection_name,
                        "embedding_model_dims": config.embedding_model_dims,
                        "path": config.save_dir,
                        "on_disk": True,
                    }
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": config.retriever_name_or_path,
                        "embedding_dims": config.embedding_model_dims,
                        "model_kwargs": {"device": config.use_gpu},
                    },
                }
            },
        }
        self.config = config
        self.memory_layer = Memory.from_config(self.memory_config)

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        "check if the memory config is right"
        if user_id is None:
            user_id = self.config.user_id
        try:
            _ = self.memory_layer.load(user_id=user_id, limit=1)
            return True
        except Exception as e:
            return False

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Add a message to the memory layer."""
        self.memory_layer.add(
            messages = [message],
            user_id = self.config.user_id
        )

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """Add a list of messages to the memory layer."""
        self.memory_layer.add(
            messages = messages,
            user_id = self.config.user_id
        )

    def retrive(self, query: str, k:int = 10, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieve the memories"""
        related_memories = self.memory_layer.search(query, limit = k)
        outputs = []
        for mem in related_memories.get("results", []):
            outputs.append(
                {
                    "content": mem["memory"],
                    "metadata": mem["metadata"]
                }
            )
        return outputs

    def delete(self, memory_id: str) -> None:
        """Delete a memory from the memory layer."""
        self.memory_layer.delete(memory_id)
    
    def update(self, memory_id: str, data: str) -> None:
        """Update a memory in the memory layer."""
        self.memory_layer.update(memory_id, data)


