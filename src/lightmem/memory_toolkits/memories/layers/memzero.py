from __future__ import annotations
from .base import BaseMemoryLayer 
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from .baselines.mem0 import (
    Memory,
    MemoryConfig
) 
from typing import (
    Literal, 
    List, 
    Dict, 
    Any,
    Optional, 
)
import os
import json
import pickle
import logging

logger = logging.getLogger(__name__)

class MemZeroConfig(BaseModel):
    """The default configuration for MemZero"""

    # Config for memory
    user_id: str = Field(..., description="The user id of the memory system.")

    save_dir: str = Field(
        default = "vector_store/memzero", 
        description = "The directory to save the memory."
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
        self.config = config
        self.memory_config = self._build_memory_config()

        try:
            self.memory_layer = Memory.from_config(self.memory_config)
            logger.info(f"MemZeroLayer initialized for user: {config.user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            raise RuntimeError(f"Failed to initialize Mem0: {e}") from e

    def _build_memory_config(self) -> Dict[str, Any]:
        """Build the configuration dictionary for Mem0."""
        return {
            "llm": {
                "provider": self.config.llm_backend,
                "config": {
                    "model": self.config.llm_model,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                    "openai_base_url": os.environ.get("OPENAI_API_BASE"),
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": self.config.user_id,
                    "embedding_model_dims": self.config.embedding_model_dims,
                    "path": self.config.save_dir,
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": self.config.retriever_name_or_path,
                    "embedding_dims": self.config.embedding_model_dims,
                    "model_kwargs": {"device": self.config.use_gpu},
                },
            }
        }

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """Load the memory of the user."""
        if user_id is None:
            user_id = self.config.user_id
            
        pkl_path = os.path.join(self.config.save_dir, f"{user_id}.pkl")
        config_path = os.path.join(self.config.save_dir, "config.json")
        
        if not os.path.exists(pkl_path) or not os.path.exists(config_path):
            logger.info(f"No saved memory found for user {user_id}")
            return False 
        
        try:
            with open(config_path, 'r', encoding="utf-8") as f:
                config_dict = json.load(f)
                    
            if user_id != config_dict["user_id"]:
                raise ValueError(
                    f"The user id in the config file ({config_dict['user_id']}) "
                    f"does not match the user id ({user_id}) in the function call."
                )
                
            self.config = MemZeroConfig(**config_dict)
            self.memory_config = self._build_memory_config()
            self.memory_layer = Memory.from_config(self.memory_config)
                
            with open(pkl_path, "rb") as f:
                memories_data = pickle.load(f)
                
            if memories_data:
                for memory_item in memories_data:
                    try:
                        self.memory_layer.add(
                            messages=[{"role": "user", "content": memory_item.get("memory", "")}],
                            user_id=user_id,
                            infer=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to restore memory {memory_item.get('id', 'unknown')}: {e}")
            
            logger.info(f"Successfully loaded {len(memories_data)} memories for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory for user {user_id}: {e}")
            return False

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Add a message to the memory layer."""
        self.memory_layer.add(
            messages=message['content'],
            user_id=self.config.user_id
        )

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """Add a list of messages to the memory layer."""
        self.memory_layer.add(
            messages=messages,
            user_id=self.config.user_id
        )

    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, str | Dict[str, Any]]]:
        """Retrieve the memories."""
        related_memories = self.memory_layer.search(
            query=query, 
            user_id=self.config.user_id, 
            limit=k
        )
        outputs = []
        for mem in related_memories.get("results", []):
            outputs.append(
                {
                    "content": mem["memory"],
                    "metadata": {
                        key: value
                        for key, value in mem.items() if key != "memory"
                    }
                }
            )
        return outputs

    def delete(self, memory_id: str) -> bool:
        """Delete a memory from the memory layer."""
        try:
            self.memory_layer.delete(memory_id)
            return True
        except Exception as e:
            print(f"Error in delete method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def delete_all(self) -> bool:
        """Delete all memories of the user."""
        try:
            self.memory_layer.delete_all(user_id=self.config.user_id)
            return True
        except Exception as e:
            print(f"Error in delete_all method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory in the memory layer."""
        try:
            data = kwargs.get("data", "")
            self.memory_layer.update(memory_id, data)
            return True
        except Exception as e:
            print(f"Error in update method in MemZeroLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def save_memory(self) -> None:
        """Save the memory state to storage."""
        try:
            os.makedirs(self.config.save_dir, exist_ok=True)
            self._save_config()
            
            all_memories = self.memory_layer.get_all(
                user_id=self.config.user_id,
                limit=100000
            )
            
            memories_data = self._normalize_memory_data(all_memories)
            pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
            
            with open(pkl_path, "wb") as f:
                pickle.dump(memories_data, f)
                
            logger.info(f"Successfully saved {len(memories_data)} memories for user {self.config.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving memories for user {self.config.user_id}: {e}")
            raise RuntimeError(f"Error saving memories for user {self.config.user_id}: {e}") from e

    def _save_config(self) -> None:
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            "user_id": self.config.user_id,
            "save_dir": self.config.save_dir,
            "retriever_name_or_path": self.config.retriever_name_or_path,
            "embedding_model_dims": self.config.embedding_model_dims,
            "use_gpu": self.config.use_gpu,
            "llm_backend": self.config.llm_backend,
            "llm_model": self.config.llm_model,
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_API_BASE"),
        }
        
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        
    def _normalize_memory_data(self, all_memories) -> List[Dict[str, Any]]:
        if isinstance(all_memories, dict) and "results" in all_memories:
            return all_memories["results"]
        elif isinstance(all_memories, list):
            return all_memories
        else:
            logger.warning(f"Unexpected memory data format: {type(all_memories)}")
            return []
