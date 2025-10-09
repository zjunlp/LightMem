from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Type, Any, List

class MMEmbedderConfig(BaseModel):
    model_name: str = Field(description="The multimodal embedding model or Deployment platform (e.g., 'openai', 'ollama')", default="huggingface")

    _model_list: List[str] = [
        "huggingface"
    ]

    configs: Optional[dict] = Field(description="Configuration for the specific multimodal embedding model", default={})

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        model_name = values.get('model_name', cls.model_name.default)
        if model_name not in cls._model_list:
            raise ValueError(
                f"Unsupported model: {model_name}."
            )
        return values
    