from lightmem.memory.lightmem import LightMemory
from lightmem.memory.utils import MemoryEntry


def build_config():
    return {
        "index_strategy": "hybrid",
        "retrieve_strategy": "hybrid",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dims": 384,
                "model_kwargs": {"device": "cpu"},
            },
        },
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "lightmem-bm25-demo",
                "embedding_model_dims": 384,
                "path": "/tmp/qdrant_bm25_demo",
            },
        },
        "context_retriever": {
            "model_name": "BM25",
            "configs": {
                "on_disk": False,
                "index_path": "bm25_index.pkl",
            },
        },
    }


def main():
    lm = LightMemory.from_config(build_config())

    entries = [
        MemoryEntry(memory="Apple fruit"),
        MemoryEntry(memory="Apple computer"),
        MemoryEntry(memory="Banana"),
    ]

    lm.offline_update(entries)

    results = lm.retrieve("fruit", limit=5)
    print("Query results:")
    print(results)


if __name__ == "__main__":
    main()
