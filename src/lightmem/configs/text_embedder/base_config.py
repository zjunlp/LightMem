from typing import Any, Dict, Optional

from pydantic import BaseModel


class BaseTextEmbedderConfig(BaseModel):
    model: Optional[str] = None
    api_key: Optional[str] = None
    embedding_dims: Optional[int] = None
    ollama_base_url: Optional[str] = None
    openai_base_url: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    huggingface_base_url: Optional[str] = None
