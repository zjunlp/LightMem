import os
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, Type, Any

class LlmLingua2Config(BaseModel):
    llmlingua_config: Dict[str, Any] = Field(
        default={
            "model_name": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "device_map": "cuda",
            "use_llmlingua2": True,
        },
        description="Configuration for LLMLingua, including model name, device, and whether to use LLMLingua-2."
    )

    llmlingua2_config: Dict[str, Any] = Field(
        default={
            "max_batch_size": 50,
            "max_force_token": 100,
        },
        description="Advanced configuration for LLMLingua-2 (batch size, token control)"
    )

    compress_config: Dict[str, Any] = Field(
        default={
            "instruction": "",
            "rate": 0.8,
            "target_token": -1
        },
        description="Additional instruction text to be included in the prompt, The maximum compression rate, "
    )

    @field_validator("llmlingua_config")
    @classmethod
    def validate_llmlingua_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        allowed_models = [
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "NousResearch/Llama-2-7b-hf",
            None
        ]
        model_name = v.get("model_name")

        if model_name is not None:
            if model_name not in allowed_models and not os.path.exists(model_name):
                raise ValueError(
                    f"model_name must be one of {allowed_models} "
                    f"or a valid local path (got {model_name})"
                )
        
        if "use_llmlingua2" in v and not isinstance(v["use_llmlingua2"], bool):
            raise ValueError("use_llmlingua2 must be a boolean")
        
        return v
    
    @field_validator("llmlingua2_config")
    @classmethod
    def validate_llmlingua2_config(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v.get("max_batch_size"), int) or v["max_batch_size"] <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if not isinstance(v.get("max_force_token"), int) or v["max_force_token"] <= 0:
            raise ValueError("max_force_token must be a positive integer")
        return v