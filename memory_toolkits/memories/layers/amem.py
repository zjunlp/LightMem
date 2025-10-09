from __future__ import annotations
from .base import BaseMemoryLayer 
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from .baselines.agentic_memory.memory_system import (
    AgenticMemorySystem, 
    MemoryNote, 
) 
import pickle 
import os
import json
from typing import (
    Literal, 
    List, 
    Dict, 
    Any,
    Optional, 
)

class AMEMConfig(BaseModel):
    """The default configuration for A-MEM."""

    user_id: str = Field(..., description="The user id of the memory system.")
    retriever_name_or_path: str = Field(
        default="all-MiniLM-L6-v2",
        description="The name or path of the retriever model to use.",
    )
    llm_backend: Literal["openai", "ollama"] = Field(
        default="openai",
        description="The backend to use for the LLM. Currently, only openai and ollama are supported.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="The base backbone model to use.",
    )

    # In A-MEM, each memory evolution operation modifies the keywords, tags, and context of notes. 
    # However, the corresponding embeddings are not updated. 
    # If the embeddings were updated every time a note is added, the overhead would be substantial. 
    # Therefore, A-MEM introduces a hyperparameter `evo_threshold`
    # where after adding `evo_threshold` notes, all note embeddings are updated.
    evo_threshold: int = Field(
        default=100,
        description="The threshold for the number of memories to trigger evolution.",
        gt=0,
    )

    api_key: str | None = Field(
        default=None,
        description="The API key to use for the LLM. It is used for openai backend. "
        "If not provided, the API key will be loaded from the environment variable.",
    )
    save_dir: str = Field(default="amem", description="The directory to save the memory.")

    @model_validator(mode="after")
    def _validate_save_dir(self) -> AMEMConfig:
        if os.path.isfile(self.save_dir):
            raise AssertionError(f"Provided path ({self.save_dir}) should be a directory, not a file")
        return self 

    
class AMEMLayer(BaseMemoryLayer):

    layer_type: str = "amem"

    def __init__(self, config: AMEMConfig) -> None:
        """Create an interface of A-MEM. The implemenation is based on the 
        [official implementation](https://github.com/WujiangXu/A-mem-sys)."""
        self.memory_layer = AgenticMemorySystem(
            model_name=config.retriever_name_or_path,
            llm_backend=config.llm_backend,
            llm_model=config.llm_model,
            evo_threshold=config.evo_threshold,
            api_key=config.api_key,
            user_id=config.user_id, 
        )
        self.config = config 
    
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
        self.config = AMEMConfig(**config_dict)
        self.memory_layer = AgenticMemorySystem(
            model_name=self.config.retriever_name_or_path,
            llm_backend=self.config.llm_backend,
            llm_model=self.config.llm_model,
            evo_threshold=self.config.evo_threshold,
            api_key=self.config.api_key,
            user_id=self.config.user_id,
        )
        
        with open(pkl_path, "rb") as f:
            predefined_states = pickle.load(f)
        self.memory_layer.evo_cnt = predefined_states["evo_cnt"]
        predefined_notes = predefined_states["notes"]

        documents, metadatas, ids, embeddings = [], [], [], [] 
        for note in predefined_notes:
            self.memory_layer.memories[note["id"]] = MemoryNote(
                content=note["content"],
                id=note["id"],
                keywords=note["keywords"],
                links=note["links"],
                retrieval_count=note["retrieval_count"],
                timestamp=note["timestamp"],
                last_accessed=note["last_accessed"],
                context=note["context"], 
                evolution_history=note["evolution_history"],
                category=note["category"],
                tags=note["tags"],
            )
            documents.append(note["database"]["document"])
            metadatas.append(note["database"]["metadata"])
            ids.append(note["database"]["id"])
            embeddings.append(note["database"]["embedding"])

        self.memory_layer.retriever.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        return True 
    
    def consolidate_memories(self) -> None:
        """Consolidate the memories. It can be called manually after the process of memory construction."""
        self.memory_layer.consolidate_memories()

    def add_message(self, message: Dict[str, str], **kwargs) -> None:
        """Add a message to the memory layer."""
        if "timestamp" not in kwargs: 
            raise KeyError("timestamp is required in `kwargs`")
        timestamp = kwargs["timestamp"]
        
        # See https://github.com/WujiangXu/A-mem/blob/main/test_advanced.py#L296 
        text = f"Speaker {message['role']} says: {message['content']}"
        self.memory_layer.add_note(text, time=timestamp)

    def add_messages(self, messages: List[Dict[str, str]], **kwargs) -> None:
        """Add a list of messages to the memory layer."""        
        for message in messages: 
            self.add_message(message, **kwargs)
    
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Dict[str, str | Dict[str, Any]]]:
        """Retrieve the memories."""
        memories = self.memory_layer.search_agentic(query, k=k)
        outputs = [] 
        for memory in memories:
            used_content = {
                "memory content": memory["content"], 
                "memory context": memory["context"],
                "memory keywords": str(memory["keywords"]),
                "memory tags": str(memory["tags"]),
                "talk start time": memory["timestamp"],
            }
            outputs.append(
                {
                    "content": memory["content"], 
                    "metadata": {
                        key: value
                        for key, value in memory.items() if key != "content"
                    },
                    # See https://github.com/WujiangXu/A-mem/blob/main/memory_layer.py#L690. 
                    "used_content": '\n'.join(
                        [f"{key}: {value}" for key, value in used_content.items()]
                    )
                }
            )
        return outputs 

    def delete(self, memory_id: str) -> bool:
        """Delete the memory."""
        return self.memory_layer.delete(memory_id)

    def update(self, memory_id: str, **kwargs) -> bool:
        """Update the memory."""
        return self.memory_layer.update(memory_id, **kwargs)

    def save_memory(self) -> None:
        """Save the memory to a directory with config.json and memory .pkl."""
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            "retriever_name_or_path": self.config.retriever_name_or_path,
            "llm_backend": self.config.llm_backend,
            "llm_model": self.config.llm_model,
            "evo_threshold": self.config.evo_threshold,
            "user_id": self.config.user_id,
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # Serialize notes with associated vector store entries
        notes_serialized = []
        collection = self.memory_layer.retriever.collection

        for note in self.memory_layer.memories.values():
            fetched = collection.get(
                ids=[note.id], include=["documents", "metadatas", "embeddings"]
            )
            doc_value = fetched["documents"][0]
            meta_value = fetched["metadatas"][0]
            # Note that it is a numpy array, which is picklable 
            emb_value = fetched["embeddings"][0]

            note_dict = {
                "content": note.content,
                "id": note.id,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
                "database": {
                    "document": doc_value,
                    "metadata": meta_value, 
                    "id": note.id,
                    "embedding": emb_value,
                },
            }
            notes_serialized.append(note_dict)

        payload = {
            "evo_cnt": self.memory_layer.evo_cnt,
            "notes": notes_serialized,
        }

        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)
