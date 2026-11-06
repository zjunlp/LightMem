from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Optional


class BM25Config(BaseModel):
    """Configuration for BM25 retriever."""

    k1: float = Field(1.5, description="BM25 k1 parameter controlling term frequency scaling")
    b: float = Field(0.75, description="BM25 b parameter controlling document length normalization")
    tokenizer: Optional[str] = Field(None, description="Optional tokenizer identifier or function name")
    use_stemming: Optional[bool] = Field(False, description="Whether to apply stemming on tokens")
    lowercase: Optional[bool] = Field(True, description="Whether to lowercase text before tokenization")
    on_disk: Optional[bool] = Field(False, description="Enable local cache for indexed corpus")

    @model_validator(mode="before")
    @classmethod
    def check_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        k1, b = values.get("k1"), values.get("b")
        if b is not None and not (0 < b <= 1):
          raise ValueError("Parameter 'b' must be between 0 and 1.")
        if k1 is not None and not (0 < k1):
          raise ValueError("Parameter 'k1' must be greater than 0.")

        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. "
                f"Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __repr__(self):
        return (
            f"<BM25Config(k1={self.k1}, b={self.b}, tokenizer={self.tokenizer}, "
            f"use_stemming={self.use_stemming}, lowercase={self.lowercase}, on_disk={self.on_disk})>"
        )
