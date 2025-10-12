from typing import Dict, Optional, ClassVar
from importlib import import_module
from pydantic import BaseModel, Field, model_validator

class EmbeddingRetrieverConfig(BaseModel):
    model_name: str = Field(
        description="The embedding retriever or vector store (e.g., 'qdrant', 'chroma', 'upstash_vector')",
        default="qdrant",
    )

    configs: Optional[Dict] = Field(description="Configuration for the specific vector store", default={})

    _model_list: ClassVar[Dict[str, str]] = {
        "qdrant": "lightmem.configs.retriever.embeddingretriever.qdrant.QdrantConfig"
    }

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        default_model = cls.__pydantic_fields__["model_name"].default
        model_name = values.get("model_name", default_model)

        if model_name not in cls._model_list:
            raise ValueError(f"Unsupported model: {model_name}.")

        values["model_name"] = model_name
        return values

    @model_validator(mode="after")
    def validate_and_create_config(self) -> "EmbeddingRetrieverConfig":
        config_path = self._model_list[self.model_name]
        module_path, class_name = config_path.rsplit('.', 1)
        
        try:
            module = import_module(module_path)
            config_class = getattr(module, class_name)
            if self.configs is None:
                self.configs = config_class()  
            elif isinstance(self.configs, Dict):
                self.configs = config_class(**self.configs)
                
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load config class '{config_path}': {e}")
        return self
