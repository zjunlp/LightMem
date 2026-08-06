"""Vector store abstract interface and default FAISS implementation"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np


class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add vectors with the given ids as unique identifiers"""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search nearest neighbors; return a list of (id, score)"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by id"""
        pass

    @abstractmethod
    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get the vector for the specified id"""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation"""

    def __init__(self, dimension: int):
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu package is required for FAISSVectorStore. "
                "Please install it with: pip install faiss-cpu"
            ) from exc

        self.dimension = dimension
        self._faiss = faiss
        # Use inner-product similarity (requires normalized vectors)
        self._index = faiss.IndexFlatIP(dimension)
        # Two-way mapping between id and index
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        # Store the original vectors for get_vector queries
        self._vectors: Dict[str, np.ndarray] = {}
        self._next_index = 0

    def add(self, ids: List[str], vectors: np.ndarray) -> None:
        """Add vectors with the given ids as unique identifiers"""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"Number of ids ({len(ids)}) does not match "
                f"number of vectors ({vectors.shape[0]})"
            )

        # Normalize vectors so that inner product equals cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = vectors / norms

        for i, vec_id in enumerate(ids):
            # If the id already exists, remove the old vector first
            if vec_id in self._id_to_index:
                self.delete([vec_id])

            idx = self._next_index
            self._next_index += 1
            self._id_to_index[vec_id] = idx
            self._index_to_id[idx] = vec_id
            self._vectors[vec_id] = vectors[i].copy()

        self._index.add(normalized)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Search nearest neighbors; return a list of (id, score)"""
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query_vector.shape[1]}"
            )

        # Normalize the query vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # If the index is empty, return an empty list
        if self._index.ntotal == 0:
            return []

        # Cap top_k by the number of vectors actually in the index
        actual_k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            vec_id = self._index_to_id.get(int(idx))
            if vec_id is not None:
                results.append((vec_id, float(score)))
        return results

    def delete(self, ids: List[str]) -> None:
        """Delete vectors by id"""
        # FAISS IndexFlatIP does not support deletion directly; use a rebuild strategy
        indices_to_remove = set()
        for vec_id in ids:
            if vec_id in self._id_to_index:
                idx = self._id_to_index[vec_id]
                indices_to_remove.add(idx)
                del self._id_to_index[vec_id]
                del self._index_to_id[idx]
                del self._vectors[vec_id]

        if not indices_to_remove:
            return

        # Rebuild the index: keep vectors that were not deleted
        if self._vectors:
            remaining_ids = list(self._vectors.keys())
            remaining_vectors = np.array(
                [self._vectors[vid] for vid in remaining_ids], dtype=np.float32
            )
            # Normalize
            norms = np.linalg.norm(remaining_vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = remaining_vectors / norms

            # Rebuild index and mappings
            self._index = self._faiss.IndexFlatIP(self.dimension)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._next_index = 0

            self._index.add(normalized)
            for i, vec_id in enumerate(remaining_ids):
                self._id_to_index[vec_id] = i
                self._index_to_id[i] = vec_id
            self._next_index = len(remaining_ids)
        else:
            # All vectors have been deleted; reset the index
            self._index = self._faiss.IndexFlatIP(self.dimension)
            self._id_to_index.clear()
            self._index_to_id.clear()
            self._next_index = 0

    def get_vector(self, id: str) -> Optional[np.ndarray]:
        """Get the vector for the specified id"""
        return self._vectors.get(id)

    def __len__(self) -> int:
        """Return the number of stored vectors"""
        return len(self._vectors)
