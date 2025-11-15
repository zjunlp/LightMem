from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Type, Any, List, ClassVar
from .base_config import BaseMemoryManagerConfig


class MemoryManagerConfig(BaseModel):
    model_name: str = Field(description="The memory management model or Deployment platform (e.g., 'openai', 'ollama'...)", default="openai")

    _model_list: ClassVar[List[str]] = [
        "openai",
        "deepseek",
        "transformers",
        "ollama",
        "vllm",
        "vllm_offline",
    ]

    configs: Optional[dict] = Field(description="Configuration for the specific MemoryManager model", default={})

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        default_model = cls.__pydantic_fields__["model_name"].default
        model_name = values.get("model_name", default_model)

        if model_name not in cls._model_list:
            raise ValueError(f"Unsupported model: {model_name}.")

        values["model_name"] = model_name
        return values
    
    @model_validator(mode='after')
    def load_config_class(self) -> 'MemoryManagerConfig':
        if self.configs is None:
            self.configs = BaseMemoryManagerConfig()
        elif isinstance(self.configs, dict):
            self.configs = BaseMemoryManagerConfig(**self.configs)
        
        return self