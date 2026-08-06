"""Hybrid-scoring semantic retriever

Implements the hybrid scoring formula from the paper:
    Score(v, o_t) = cosine_sim(v, o_t) + BM25(v, o_t) + LLM_ver(v, o_t)

Fuses three signals:
- dense embedding cosine similarity (via BaseEmbedder + BaseVectorStore)
- BM25 sparse retrieval score (TF-IDF variant, k1=1.5, b=0.75)
- LLM verification score (via BaseLLM.verify())
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..graph.nodes import SemanticNode
from ..interfaces.embedder import BaseEmbedder
from ..interfaces.llm import BaseLLM
from ..interfaces.vectorstore import BaseVectorStore
from .base import BaseRetriever


class SemanticRetriever(BaseRetriever):
    """Hybrid-scoring semantic retriever

    Ranks SemanticNode candidates by fusing three signals: dense retrieval
    (cosine similarity), sparse retrieval (BM25), and LLM verification.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.5,
        llm_weight: float = 0.3,
    ):
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self.vectorstore = vectorstore
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.llm_weight = llm_weight

        # BM25 parameters
        self._k1 = 1.5
        self._b = 0.75

        # BM25 index state
        self._doc_freqs: Counter = Counter()       # Number of documents containing each term
        self._doc_tokens: Dict[str, List[str]] = {}  # node_id -> token list
        self._doc_lengths: Dict[str, int] = {}      # node_id -> document length
        self._avg_dl: float = 0.0                   # Average document length
        self._num_docs: int = 0                     # Total number of documents
        self._idf_cache: Dict[str, float] = {}      # token -> IDF value

    # ==================== Public interface ====================

    async def retrieve(self, query: str, top_k: int = 5) -> List[SemanticNode]:
        """Retrieve relevant semantic nodes based on the query.

        Use the hybrid scoring formula to rank candidate nodes and return the top-k results.

        Args:
            query: Query text
            top_k: Maximum number of nodes to return

        Returns:
            List of SemanticNode sorted by relevance in descending order
        """
        # 1. Get dense retrieval candidates from the vector store
        query_embedding = await self.embedder.embed_text(query)
        dense_results = self.vectorstore.search(query_embedding, top_k=top_k * 3)
        # dense_results: List[Tuple[node_id, cosine_score]]

        if not dense_results:
            return []

        candidate_ids = [rid for rid, _ in dense_results]
        dense_scores: Dict[str, float] = {rid: score for rid, score in dense_results}

        # 2. Collect candidate nodes
        candidates: List[SemanticNode] = []
        id_to_node: Dict[str, SemanticNode] = {}
        for nid in candidate_ids:
            node = self.graph.semantic_nodes.get(nid)
            if node is not None:
                candidates.append(node)
                id_to_node[nid] = node

        if not candidates:
            return []

        # 3. Compute BM25 scores
        bm25_scores = self._compute_bm25(query, [n.content for n in candidates])
        bm25_map: Dict[str, float] = {}
        for node, score in zip(candidates, bm25_scores):
            bm25_map[node.id] = score

        # 4. Compute LLM verification scores
        llm_scores = await self._compute_llm_verification(query, candidates)
        llm_map: Dict[str, float] = {}
        for node, score in zip(candidates, llm_scores):
            llm_map[node.id] = score

        # 5. Normalize each score to [0, 1] then fuse with weights
        all_dense = list(dense_scores.values())
        all_bm25 = list(bm25_map.values())
        all_llm = list(llm_map.values())

        norm_dense = self._normalize_scores(all_dense)
        norm_bm25 = self._normalize_scores(all_bm25)
        norm_llm = self._normalize_scores(all_llm)

        # 6. Compute the weighted total score and sort
        scored_nodes: List[Tuple[SemanticNode, float]] = []
        for i, node in enumerate(candidates):
            total = (
                self.dense_weight * norm_dense[i]
                + self.bm25_weight * norm_bm25[i]
                + self.llm_weight * norm_llm[i]
            )
            scored_nodes.append((node, total))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored_nodes[:top_k]]

    def build_index(self) -> None:
        """Build/rebuild the vector index and BM25 index from the graph"""
        # --- Rebuild the vector index ---
        nodes = list(self.graph.semantic_nodes.values())
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

        # --- Rebuild the BM25 index ---
        self._doc_freqs = Counter()
        self._doc_tokens = {}
        self._doc_lengths = {}
        self._idf_cache = {}
        self._num_docs = len(nodes)

        total_length = 0
        for node in nodes:
            tokens = self._tokenize(node.content)
            self._doc_tokens[node.id] = tokens
            self._doc_lengths[node.id] = len(tokens)
            total_length += len(tokens)
            # Track whether each term occurs in the document (used to compute df)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

        self._avg_dl = total_length / self._num_docs if self._num_docs > 0 else 0.0

        # Precompute IDF
        for token, df in self._doc_freqs.items():
            self._idf_cache[token] = math.log(
                (self._num_docs - df + 0.5) / (df + 0.5) + 1.0
            )

    # ==================== BM25 ====================

    def _compute_bm25(self, query: str, documents: List[str]) -> List[float]:
        """Compute BM25 scores.

        Uses the classic Okapi BM25 formula:
            BM25(D, Q) = Σ_{q ∈ Q} IDF(q) * (f(q,D) * (k1+1)) / (f(q,D) + k1*(1-b+b*|D|/avgdl))

        Args:
            query: Query text
            documents: List of candidate document contents

        Returns:
            List of BM25 scores with the same length as documents
        """
        if not documents or self._num_docs == 0:
            return [0.0] * len(documents)

        query_tokens = self._tokenize(query)
        scores: List[float] = []

        for doc in documents:
            doc_tokens = self._tokenize(doc)
            doc_len = len(doc_tokens)
            tf_counter = Counter(doc_tokens)

            score = 0.0
            for qt in query_tokens:
                # Use precomputed IDF; for tokens not in the index, use a smoothed IDF
                idf = self._idf_cache.get(qt)
                if idf is None:
                    # For out-of-vocabulary tokens, use a smoothed IDF
                    idf = math.log(
                        (self._num_docs + 0.5) / (0.5 + 0.5) + 1.0
                    )

                tf = tf_counter.get(qt, 0)
                # BM25 core formula
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (
                    1 - self._b + self._b * doc_len / max(self._avg_dl, 1e-8)
                )
                score += idf * numerator / denominator

            scores.append(score)

        return scores

    # ==================== LLM Verification ====================

    async def _compute_llm_verification(
        self, query: str, candidates: List[SemanticNode]
    ) -> List[float]:
        """Use the LLM to verify the relevance of candidate nodes.

        Calls BaseLLM.verify() for each candidate node to obtain a 0-1 relevance score.

        Args:
            query: Query text
            candidates: List of candidate semantic nodes

        Returns:
            List of LLM verification scores with the same length as candidates
        """
        scores: List[float] = []
        for node in candidates:
            try:
                score = await self.llm.verify(
                    claim=query,
                    evidence=node.content,
                )
                scores.append(score)
            except Exception:
                # Give a low score on LLM call failure
                scores.append(0.0)
        return scores

    # ==================== Utility methods ====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization: lowercase and extract alphanumeric tokens.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return re.findall(r"\w+", text.lower())

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        """Min-max normalize scores to the [0, 1] range.

        If all scores are equal, return a list of zeros.

        Args:
            scores: Original score list

        Returns:
            Normalized score list
        """
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        rng = max_s - min_s
        if rng < 1e-8:
            return [0.0] * len(scores)
        return [(s - min_s) / rng for s in scores]