from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Any, List, Union, ClassVar
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig

class TextEmbedderConfig(BaseModel):
    model_name: str = Field(
        default="huggingface",
        description="The embedding model or Deployment platform (e.g., 'openai', 'huggingface')"
    )

    _model_list: ClassVar[List[str]] = ["huggingface", "openai"]

    configs: Optional[Union[BaseTextEmbedderConfig, Dict[str, Any]]] = Field(
        default=None,
        description="Configuration for the specific embedding model"
    )

    @model_validator(mode="before")
    def validate_model_name(cls, values):
        model_name = values.get("model_name", "huggingface")
        if model_name not in cls._model_list:
            raise ValueError(f"Unsupported model: {model_name}.")
        values["model_name"] = model_name
        return values

    @model_validator(mode="after")
    def load_config_class(self) -> "TextEmbedderConfig":
        if isinstance(self.configs, dict):
            self.configs = BaseTextEmbedderConfig(**self.configs)
        elif self.configs is None:
            self.configs = BaseTextEmbedderConfig()
        return self