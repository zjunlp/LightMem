from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Type, Any, List, ClassVar

class TopicSegmenterConfig(BaseModel):
    model_name: str = Field(description="The TopicSegmenter model or Deployment platform (e.g., 'openai', 'deepseek', 'ollama', 'llmlingua-2')", default="llmlingua-2")

    _model_list: ClassVar[List[str]] = [
        "llmlingua-2"
    ]

    configs: Optional[dict] = Field(description="Configuration for the specific TopicSegmenter model", default={})

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        default_model = cls.__pydantic_fields__["model_name"].default
        model_name = values.get("model_name", default_model)

        if model_name not in cls._model_list:
            raise ValueError(f"Unsupported model: {model_name}.")

        values["model_name"] = model_name
        return values
