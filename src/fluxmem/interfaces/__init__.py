"""FluxMem abstract interface layer - exports all interface classes"""
from fluxmem.interfaces.llm import BaseLLM, OpenAILLM
from fluxmem.interfaces.embedder import BaseEmbedder, OpenAIEmbedder
from fluxmem.interfaces.vectorstore import BaseVectorStore, FAISSVectorStore

__all__ = [
    "BaseLLM",
    "OpenAILLM",
    "BaseEmbedder",
    "OpenAIEmbedder",
    "BaseVectorStore",
    "FAISSVectorStore",
]
