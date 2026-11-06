from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, Type, Any, List

class ContextRetrieverConfig(BaseModel):
    model_name: str = Field(description="The Context retriever or algorithm (e.g., 'BM25', 'TF-IDF')", default="BM25")

    _model_list: List[str] = [
        "BM25"
    ]

    configs: Optional[dict] = Field(description="Configuration for the context retriever or algorithm", default={})

    @model_validator(mode='before')
    def validate_model_name(cls, values):
        model_name = values.get('model_name', cls.model_name.default)
        if model_name not in cls._model_list:
            raise ValueError(
                f"Unsupported model: {model_name}."
            )
        return values
    