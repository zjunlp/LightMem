import logging
from types import SimpleNamespace

from qdrant_client import QdrantClient, models

from lightmem.memory.lightmem import LightMemory
from lightmem.memory.utils import (
    assign_sequence_numbers_with_timestamps,
    convert_extraction_results_to_memory_entries,
)


def test_source_metadata_is_exported_to_qdrant(tmp_path):
    def msg(role, dia_id=None):
        return {"role": role, "dia_id": dia_id, "session_time": "2023-01-23 16:06:00",
                "weekday": "Mon", "speaker_name": "Deborah", "speaker_id": "speaker_a"}

    batches = [[[msg("user", "D1:1"), msg("assistant")]],
               [[msg("user", "D1:2"), msg("assistant")]]]
    batches, timestamps, weekdays, speakers, topics = assign_sequence_numbers_with_timestamps(
        batches, topic_id_mapping=[[0], [1]]
    )
    # IDs in the extraction prompt are based on the global sequence number.
    results = [None, {"cleaned_result": [[
        {"source_id": 1, "relation": "A relation"},
        {"source_id": 0, "fact": "An invalid cross-batch anchor"},
    ]]}]
    entries = convert_extraction_results_to_memory_entries(
        results, timestamps, weekdays, speakers, topics, [0, 1],
        logging.getLogger(__name__), valid_source_ids=[{0}, {1}]
    )
    assert [(e.entry_type, e.anchor_dia_id) for e in entries] == [
        ("relational", "D1:2"), ("factual", None)
    ]

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    client.create_collection("conv-48", vectors_config=models.VectorParams(
        size=2, distance=models.Distance.COSINE
    ))

    class Retriever:
        def exists(self, point_id):
            return bool(client.retrieve("conv-48", [point_id]))

        def insert(self, vectors, payloads, ids):
            client.upsert("conv-48", points=[
                models.PointStruct(id=i, vector=v, payload=p)
                for i, v, p in zip(ids, vectors, payloads)
            ])

    memory = LightMemory.__new__(LightMemory)
    memory.config = SimpleNamespace(index_strategy="embedding")
    memory.logger = logging.getLogger(__name__)
    memory.text_embedder = SimpleNamespace(embed=lambda text: [0.1, 0.2])
    memory.embedding_retriever = Retriever()
    memory.offline_update(entries)
    points, _ = client.scroll("conv-48", with_payload=True, with_vectors=False)
    payloads = {p.payload["memory"]: p.payload for p in points}
    assert payloads["A relation"]["entry_type"] == "relational"
    assert payloads["A relation"]["anchor_dia_id"] == "D1:2"
    assert payloads["An invalid cross-batch anchor"]["anchor_dia_id"] is None
    client.close()
