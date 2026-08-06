"""Embedding abstract interface and default OpenAI implementation"""
from abc import ABC, abstractmethod
from typing import List, Optional
import os
import numpy as np


class BaseEmbedder(ABC):
    """Abstract base class for embedding models"""

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text into a vector"""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding; return an (N, dim) array"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension"""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embedding implementation"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAIEmbedder. "
                "Please install it with: pip install openai"
            ) from exc

        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = AsyncOpenAI(api_key=self.api_key)
        # text-embedding-3-small outputs 1536 dimensions by default
        self._dimension = 1536

    @property
    def dimension(self) -> int:
        """Return the vector dimension"""
        return self._dimension

    async def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text into a vector"""
        response = await self._client.embeddings.create(
            model=self.model,
            input=text,
        )
        embedding = response.data[0].embedding
        # Update the actual dimension (may vary across models)
        self._dimension = len(embedding)
        return np.array(embedding, dtype=np.float32)

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch embedding; return an (N, dim) array"""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        response = await self._client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to ensure consistent order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]
        # Update the actual dimension
        if embeddings:
            self._dimension = len(embeddings[0])
        return np.array(embeddings, dtype=np.float32)
