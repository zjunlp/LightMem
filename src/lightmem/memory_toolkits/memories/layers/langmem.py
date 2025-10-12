from __future__ import annotations
from .base import BaseMemoryLayer 
from langgraph.store.memory import InMemoryStore
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from .baselines.langmem import create_memory_store_manager

from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
    field_validator,
)
from copy import deepcopy 
import pickle 
import os
import json
from typing import (
    List, 
    Dict, 
    Any,
    Optional, 
) 

class LangMemConfig(BaseModel):
    """The default configuration for LangMem."""

    user_id: str = Field(..., description="The user id of the memory system.")
    retriever_name_or_path: str = Field(
        default="huggingface:all-MiniLM-L6-v2",
        description="The name or path of the retriever model to use. "
        "The format should be `<provider>:<model_name>` where `<provider>` is one of `huggingface`, `openai`, `ollama`, etc. "
        "and `<model_name>` is the name of the model to use. "
        "For example, `huggingface:all-MiniLM-L6-v2` is the name of the all-MiniLM-L6-v2 model on Hugging Face.",
    )
    retriever_dim: int = Field(
        default=384,
        ge=1, 
        description="The dimension of the retriever model. "
        "The default value is 384, which is the dimension of the all-MiniLM-L6-v2 model on Hugging Face. "
        "If you changes the value of `retriever_name_or_path`, "
        "you need to change the value of `retriever_dim` to the dimension of the new model.",
    )
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description="The base backbone model to use. "
        "The format should be `<provider>:<model_name>` where `<provider>` is one of `openai`, `ollama`, etc. "
        "and `<model_name>` is the name of the model to use. "
        "For example, `openai:gpt-4o-mini` is the name of the gpt-4o-mini model on OpenAI.", 
    )

    save_dir: str = Field(default="langmem", description="The directory to save the memory.")

    # You can look up the following parameters in the `create_memory_store_manager` function.
    query_model: str | None = Field(
        default=None, 
        description="The model to use for generating queries. "
        "If not provided, the dialated window trick over the conversation "
        "is used to generate queries and the number of queries is controlled by `query_limit`. "
        "The format should be `<provider>:<model_name>` where `<provider>` is one of `openai`, `ollama`, etc. "
        "and `<model_name>` is the name of the model to use. "
        "For example, `openai:gpt-4o-mini` is the name of the gpt-4o-mini model on OpenAI.", 
    )
    enable_inserts: bool = Field(
        default=True, 
        description="Whether to allow creating new memory entries. "
        "When False, the manager will only update existing memories. Defaults to True.",
    )
    enable_deletes: bool = Field(
        default=True, 
        description="Whether to allow deleting existing memories "
        "that are outdated or contradicted by new information. Defaults to True.", 
    )

    # Before the agent needs to extract valuable information to be memorized
    # it firsts generate a list of queries to retrieve relevant memories.
    # The `query_limit` is the maximum number of related memories to retrieve for each query.
    # When `query_model` is not provided, the dialated window trick over the conversation 
    # is used to generate queries and the number of queries is controlled by `query_limit` (at most `query_limit // 4`).
    query_limit: int = Field(
        default=5,
        ge=1, 
        description="Maximum number of relevant memories to retrieve " 
        "for each conversation. Higher limits provide more context but may slow down processing. "
        "Defaults to 5.",
    )

    @field_validator("retriever_name_or_path", "llm_model", "query_model")
    @classmethod
    def _validate_provider_model(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if ':' not in v:
            raise ValueError("Must be in format '<provider>:<model_name>' (missing ':').")
        provider, model_name = v.split(':', 1)
        if not provider or not model_name:
            raise ValueError("Provider and model name must be non-empty, separated by ':'.")
        return v
    
    @model_validator(mode="after")
    def _validate_save_dir(self) -> LangMemConfig:
        if os.path.isfile(self.save_dir):
            raise AssertionError(f"Provided path ({self.save_dir}) should be a directory, not a file")
        return self 

class LangMemLayer(BaseMemoryLayer):

    layer_type: str = "langmem"

    def __init__(self, config: LangMemConfig) -> None:
        """Create an interface of LangMem. The implemenation is based on the 
        third-party library `langmem`."""
        self._llm_model = init_chat_model(config.llm_model)
        self._query_model = (
            None
            if config.query_model is None
            else init_chat_model(config.query_model)
        )
        self._store = InMemoryStore(
            index={
                "dims": config.retriever_dim, 
                "embed": config.retriever_name_or_path, 
                "fields": ["content"],   # `kind` is ignored as there is only one kind of memory. 
            }
        )
        self.memory_layer = create_memory_store_manager(
            self._llm_model,
            enable_inserts=config.enable_inserts, 
            enable_deletes=config.enable_deletes, 
            query_model=self._query_model,  
            query_limit=config.query_limit, 
            namespace=("memories", config.user_id),  
            store=self._store, 
        )
        self.config = config 

        # Store each memory unit's id 
        self._memory_ids = {}  
    
    @property
    def llm_model(self) -> BaseChatModel:
        """Get the LLM backbone model."""
        return self._llm_model

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Add a message to the memory layer."""
        if "timestamp" not in kwargs: 
            raise KeyError("timestamp is required in `kwargs`")
        timestamp = kwargs["timestamp"] 
        message_copy = deepcopy(message)
        message_copy["content"] = f"{message_copy['content']}\nTimestamp: {timestamp}"
        # See https://langchain-ai.github.io/langmem/background_quickstart/
        # `kwargs` can include some optional parameters, e.g., `max_steps`.
        final_puts = self.memory_layer.invoke({"messages": [message_copy]}, **kwargs)
        # Some operations update contents of previous memory units. 
        for final_put in final_puts: 
            self._memory_ids[final_put["key"]] = final_put["value"]

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """Add a list of messages to the memory layer."""    
        message_level = kwargs.pop("message_level", True)
        if message_level not in [True, False]:
            raise TypeError(
                "`message_level` must be a boolean to indicate whether the messages " 
                "are added to the memory layer message by message or as a whole."
            )
        
        if message_level:
            for message in messages: 
                self.add_message(message, **kwargs)
        else:
            final_puts = self.memory_layer.invoke({"messages": messages}, **kwargs)
            for final_put in final_puts: 
                self._memory_ids[final_put["key"]] = final_put["value"]
    
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, str | Dict[str, Any]]]:
        """Retrieve the memories."""
        memories = self.memory_layer.search(query=query, limit=k, **kwargs)
        outputs = [] 
        for memory in memories:
            memory_dict = memory.dict()
            outputs.append(
                {
                    "content": memory_dict["value"]["content"], 
                    "metadata": {
                        key: value
                        for key, value in memory_dict.items() if key != "value"
                    }, 
                    "used_content": memory_dict["value"]["content"]
                }
            )
        return outputs  

    def delete(self, memory_id: str) -> bool:
        """Delete the memory."""
        try:
            self.memory_layer.delete(memory_id)
            if memory_id in self._memory_ids:
                del self._memory_ids[memory_id]
            return True
        except Exception as e:
            print(f"Error in deleted method in LangMemLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update the memory."""
        if "content" not in kwargs:
            raise KeyError("`content` is required in `kwargs`.")
        content = kwargs.pop("content")
        try:
            self.memory_layer.put(
                memory_id, 
                {"content": content}, 
                **kwargs
            )
            return True
        except Exception as e:
            print(f"Error in update method in LangMemLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def load_memory(self, user_id: Optional[str] = None) -> bool:
        """Load the memory of the user.""" 
        if user_id is None:
            user_id = self.config.user_id
        pkl_path = os.path.join(self.config.save_dir, f"{user_id}.pkl")
        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(pkl_path) or not os.path.exists(config_path):
            return False 
        
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = json.load(f)
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )
        config = LangMemConfig(**config_dict)
        self._llm_model = init_chat_model(config.llm_model)
        self._query_model = (
            None
            if config.query_model is None
            else init_chat_model(config.query_model)
        )
        self._store = InMemoryStore(
            index={
                "dims": config.retriever_dim, 
                "embed": config.retriever_name_or_path, 
                "fields": ["content"],   
            }
        )
        self.memory_layer = create_memory_store_manager(
            self._llm_model,
            enable_inserts=config.enable_inserts, 
            enable_deletes=config.enable_deletes, 
            query_model=self._query_model,  
            query_limit=config.query_limit, 
            namespace=("memories", config.user_id),  
            store=self._store, 
        )
        self.config = config 
        
        with open(pkl_path, "rb") as f:
            predefined_memory_units = pickle.load(f)
        self._memory_ids.clear()   

        for memory_unit in predefined_memory_units:
            self.memory_layer.put(**memory_unit) 
            self._memory_ids[memory_unit["key"]] = memory_unit["value"]

        return True 

    def save_memory(self) -> None:
        """Save the memory to a directory with config.json and memory .pkl."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump()
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # In LangMem, we don't store the vector embeddings. 
        preserved_memory_units = [] 
        for key, value in self._memory_ids.items(): 
            # Note that some memory units have been deleted. 
            if self.memory_layer.get(key) is not None:
                memory_unit = {
                    "key": key,
                    "value": value,
                }
                preserved_memory_units.append(memory_unit)

        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(preserved_memory_units, f)