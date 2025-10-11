import logging
import os
import shutil

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Qdrant:
    def __init__(
        self, config: Optional[QdrantConfig] = None
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            collection_name (str): Name of the collection.
            embedding_model_dims (int): Dimensions of the embedding model.
            client (QdrantClient, optional): Existing Qdrant client instance. Defaults to None.
            host (str, optional): Host address for Qdrant server. Defaults to None.
            port (int, optional): Port for Qdrant server. Defaults to None.
            path (str, optional): Path for local Qdrant database. Defaults to None.
            url (str, optional): Full URL for Qdrant server. Defaults to None.
            api_key (str, optional): API key for Qdrant server. Defaults to None.
            on_disk (bool, optional): Enables persistent storage. Defaults to False.
        """
        if config.client:
            self.client = config.client
        else:
            params = {}
            if config.api_key:
                params["api_key"] = config.api_key
            if config.url:
                params["url"] = config.url
            if config.host and config.port:
                params["host"] = config.host
                params["port"] = config.port
            if not params:
                params["path"] = config.path
                if not config.on_disk:
                    if os.path.exists(config.path) and os.path.isdir(config.path):
                        shutil.rmtree(config.path)

            self.client = QdrantClient(**params)

        self.collection_name = config.collection_name
        self.embedding_model_dims = config.embedding_model_dims
        self.on_disk = config.on_disk
        self.create_col(config.embedding_model_dims, config.on_disk)

    def create_col(self, vector_size: int, on_disk: bool, distance: Distance = Distance.COSINE):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        response = self.list_cols()
        for collection in response.collections:
            if collection.name == self.collection_name:
                logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance, on_disk=on_disk),
        )

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        points = [
            PointStruct(
                id=idx if ids is None else ids[idx],
                vector=vector,
                payload=payloads[idx] if payloads else {},
            )
            for idx, vector in enumerate(vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def _create_filter(self, filters: dict) -> Filter:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict):
                gte = value.get("gte", None)
                lte = value.get("lte", None)
                conditions.append(FieldCondition(key=key, range=Range(gte=gte, lte=lte)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query_vector: list,
        limit: int = 5,
        filters: dict = None,
        return_full: bool = False,
    ) -> list:
        """
        Search for similar vectors.

        Args:
            query_vector (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            return_full (bool, optional): If True, return full ScoredPoint info (id, score, payload, vector).
                                        If False, return simplified dict with id and score. Defaults to False.

        Returns:
            list: Search results.
        """
        query_filter = self._create_filter(filters) if filters else None
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True, 
        )

        results = []
        for h in hits.points:
            if return_full:
                results.append({
                    "id": h.id,
                    "score": h.score,
                    "payload": h.payload,
                })
            else:
                results.append({
                    "id": h.id,
                    "score": h.score,
                })
        return results

    def delete(self, vector_id: int):
        """
        Delete a vector by ID.

        Args:
            vector_id (int): ID of the vector to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=[vector_id],
            ),
        )

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        update_data = {}
        if vector is not None:
            update_data["vector"] = vector
        if payload is not None:
            update_data["payload"] = payload

        if not update_data:
            return

        point = PointStruct(id=vector_id, **update_data)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: int) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (int): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        result = self.client.retrieve(collection_name=self.collection_name, ids=[vector_id], with_payload=True)
        return result[0] if result else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result

    def reset(self):
        """Reset the index by deleting and recreating it."""
        logger.warning(f"Resetting index {self.collection_name}...")
        self.delete_col()
        self.create_col(self.embedding_model_dims, self.on_disk)

    def exists(self, vector_id: str) -> bool:
        """
        Check if a vector with the given ID exists in the collection.

        Args:
            vector_id (str): ID of the vector to check.

        Returns:
            bool: True if the vector exists, False otherwise.
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=False,
                with_vectors=False,
            )
            return len(result) > 0
        except Exception as e:
            logger.error(f"Error checking existence of ID {vector_id}: {e}")
            return False
        
    def get_all(self, with_vectors: bool = True, with_payload: bool = True) -> list:
        """
        Retrieve all points from the collection.

        Args:
            with_vectors (bool): Whether to include vectors. Defaults to True.
            with_payload (bool): Whether to include payload. Defaults to True.

        Returns:
            list: List of all points with their vectors and payloads.
        """
        all_points = []
        offset = None
        while True:
            result, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=100, 
                with_payload=with_payload,
                with_vectors=with_vectors,
                offset=offset,
            )
            all_points.extend([p.model_dump() for p in result])
            if offset is None: 
                break
        return all_points

