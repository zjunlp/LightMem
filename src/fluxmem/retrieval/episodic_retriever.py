"""Embedding similarity-based episodic retriever

Implements the episodic retrieval formula from the paper:
    V_epi_t = TopK_{u ∈ V_epi} cos(u, o_t)

Ranks EpisodicNode candidates via dense embedding cosine similarity.
"""
from typing import Dict, List

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..graph.nodes import EpisodicNode
from ..interfaces.embedder import BaseEmbedder
from ..interfaces.vectorstore import BaseVectorStore
from .base import BaseRetriever


class EpisodicRetriever(BaseRetriever):
    """Embedding similarity-based episodic retriever

    Uses BaseEmbedder to compute the dense embedding of the query and
    EpisodicNode, performs nearest-neighbor search via BaseVectorStore,
    and returns the nodes with the highest cosine similarity.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
    ):
        self.graph = graph
        self.embedder = embedder
        self.vectorstore = vectorstore

    async def retrieve(self, query: str, top_k: int = 5) -> List[EpisodicNode]:
        """Retrieve relevant episodic nodes based on the query.

        Rank episodic nodes by cosine similarity and return the top-k results.

        Args:
            query: Query text
            top_k: Maximum number of nodes to return

        Returns:
            List of EpisodicNode sorted by relevance in descending order
        """
        # 1. Encode the query as a vector
        query_embedding = await self.embedder.embed_text(query)

        # 2. Perform nearest-neighbor search in the vector store
        search_results = self.vectorstore.search(query_embedding, top_k=top_k)
        # search_results: List[Tuple[node_id, cosine_score]]

        if not search_results:
            return []

        # 3. Get the corresponding EpisodicNode instances from the graph
        results: List[EpisodicNode] = []
        for node_id, score in search_results:
            node = self.graph.episodic_nodes.get(node_id)
            if node is not None:
                results.append(node)

        return results

    def build_index(self) -> None:
        """Build/rebuild the episodic vector index from the graph.

        Adds all EpisodicNode instances that have an embedding to the vector store.
        """
        nodes = list(self.graph.episodic_nodes.values())
        if not nodes:
            return

        # Collect nodes that have an embedding
        ids_with_emb: List[str] = []
        embeddings: List[np.ndarray] = []
        for node in nodes:
            if node.embedding is not None:
                ids_with_emb.append(node.id)
                embeddings.append(node.embedding)

        if ids_with_emb:
            emb_matrix = np.stack(embeddings)
            # Normalize for cosine similarity
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            emb_matrix = emb_matrix / norms
            self.vectorstore.add(ids_with_emb, emb_matrix)