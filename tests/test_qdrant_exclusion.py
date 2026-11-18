import copy
import logging

import pytest
from qdrant_client import QdrantClient

from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.memory.utils import retrieve_supplementary_entries


@pytest.fixture
def store():
    client = QdrantClient(location=":memory:")
    retriever = Qdrant(
        QdrantConfig(
            client=client,
            path=":memory:",
            collection_name="exclusion_test",
            embedding_model_dims=2,
        )
    )
    retriever.insert(
        vectors=[[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
        ids=[1, 2, 3],
        payloads=[
            {
                "memory": "current event",
                "category": "personal",
                "time_stamp": "t1",
                "float_time_stamp": 10,
            },
            {
                "memory": "related event",
                "category": "personal",
                "time_stamp": "t2",
                "float_time_stamp": 20,
            },
            {
                "memory": "other event",
                "category": "work",
                "time_stamp": "t3",
                "float_time_stamp": 30,
            },
        ],
    )
    yield retriever
    client.close()


def test_excludes_point_ids_without_payload_id(store):
    results = store.search([1.0, 0.0], limit=3, exclude_ids=[1])

    assert [result["id"] for result in results] == [2, 3]


def test_payload_id_does_not_control_point_exclusion(store):
    store.client.set_payload(
        collection_name=store.collection_name, payload={"id": 99}, points=[1]
    )
    store.client.set_payload(
        collection_name=store.collection_name, payload={"id": 1}, points=[2]
    )

    results = store.search([1.0, 0.0], limit=3, exclude_ids=[1], return_full=True)

    assert [result["id"] for result in results] == [2, 3]
    assert results[0]["payload"]["id"] == 1


def test_combines_metadata_filters_and_exclusion_without_mutating_inputs(store):
    filters = {"category": "personal", "float_time_stamp": {"gte": 10, "lte": 20}}
    original_filters = copy.deepcopy(filters)
    excluded = [1]

    results = store.search(
        [1.0, 0.0], limit=3, filters=filters, exclude_ids=excluded, return_full=True
    )

    assert [result["id"] for result in results] == [2]
    assert results[0]["payload"]["category"] == "personal"
    assert results[0]["score"] == pytest.approx(0.9701425)
    assert filters == original_filters
    assert excluded == [1]


@pytest.mark.parametrize("excluded", [None, []])
@pytest.mark.parametrize("return_full", [False, True])
def test_empty_exclusion_preserves_limit_scores_and_result_shape(
    store, excluded, return_full
):
    results = store.search(
        [1.0, 0.0],
        limit=1,
        filters={"category": "personal"},
        exclude_ids=excluded,
        return_full=return_full,
    )

    assert [result["id"] for result in results] == [1]
    assert results[0]["score"] == pytest.approx(1.0)
    assert set(results[0]) == (
        {"id", "score", "payload"} if return_full else {"id", "score"}
    )


def test_excluding_every_point_returns_empty_results(store):
    assert store.search([1.0, 0.0], limit=3, exclude_ids=[1, 2, 3]) == []


def test_excludes_uuid_point_ids(store):
    ids = [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    store.client.delete_collection(store.collection_name)
    store.create_col(vector_size=2, on_disk=False)
    store.insert(vectors=[[1.0, 0.0], [0.0, 1.0]], payloads=[{}, {}], ids=ids)

    results = store.search([1.0, 0.0], limit=2, exclude_ids=[ids[0]])

    assert [result["id"] for result in results] == [ids[1]]


@pytest.mark.parametrize("additional_filters", [None, {"category": "personal"}])
def test_supplementary_retrieval_does_not_return_current_buffer(
    store, additional_filters
):
    class FixedEmbedder:
        def embed(self, text):
            return [1.0, 0.0]

    buffer_entries = [entry for entry in store.get_all() if entry["id"] == 1]
    results = retrieve_supplementary_entries(
        buffer_entries=buffer_entries,
        retriever=store,
        text_embedder=FixedEmbedder(),
        top_k=1,
        additional_filters=additional_filters,
        logger=logging.getLogger(__name__),
    )

    assert [result["id"] for result in results] == [2]
    assert results[0]["payload"]["memory"] == "related event"
