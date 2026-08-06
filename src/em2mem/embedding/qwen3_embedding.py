# Derived from an external implementation; see LICENSE for attribution and upstream license terms.
# Changes made by anonymous authors

import importlib.util
import os
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = device

        model_kwargs = {"dtype": "auto", "device_map": device}
        attn_implementation = os.getenv("EM2MEM_TEXT_ATTN_IMPLEMENTATION", "").strip()
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        elif importlib.util.find_spec("flash_attn") is not None:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = SentenceTransformer(
            model_name,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
