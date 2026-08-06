"""Retriever abstract base class"""
from abc import ABC, abstractmethod
from typing import List

from ..graph.nodes import BaseNode


class BaseRetriever(ABC):
    """Abstract base class for retrievers, defining a unified retrieval interface"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[BaseNode]:
        """Retrieve relevant nodes based on the query.

        Args:
            query: Query text
            top_k: Maximum number of nodes to return

        Returns:
            List of nodes sorted by relevance in descending order
        """
        pass
