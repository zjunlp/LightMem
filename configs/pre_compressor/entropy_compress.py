import os
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, Any


class EntropyCompressorConfig(BaseModel):
    entropy_config: Dict[str, Any] = Field(
        default={
            "model_name": "gpt2",
            "device": "cuda",
            "word_level_strategy": "average",  # or "first_token"
            "compress_rate": 0.5,  
            "max_length": 512
        },
        description="Configuration for entropy-based semantic unit compression."
    )

    @field_validator("entropy_config")
    @classmethod
    def validate_entropy_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v.get("model_name"), str):
            raise ValueError("model_name must be a string")
        if v.get("word_level_strategy") not in ["average", "first_token"]:
            raise ValueError("word_level_strategy must be 'average' or 'first_token'")
        if not (0 < v.get("compress_rate") <= 1.0):
            raise ValueError("compress_rate must be between 0 and 1.0")
        return v
