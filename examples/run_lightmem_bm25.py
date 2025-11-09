"""
Example: Demonstrate basic usage of LightMemory with BM25 retrieval.
This script follows the same structure and annotation style as other examples in LightMem.
"""

import os
from lightmem.memory.lightmem import LightMemory


# ============ Data Configuration ============
EXAMPLE_COLLECTION = "demo_collection_bm25"


def load_lightmem_bm25(collection_name):
    """
    Load a LightMemory instance configured to use BM25 retrieval.
    This function mirrors the config structure in other examples.
    """
    config = {
        "pre_compress": False,
        "topic_segment": False,
        "index_strategy": "bm25",  # ✅ Use BM25 instead of embedding
        "retrieve_strategy": "bm25",
        "bm25_retriever": {
            "model_name": "bm25",
            "configs": {
                "collection_name": collection_name,
                "path": f"./bm25_data/{collection_name}",
            },
        },
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 2048,
            }
        },
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": "logs",
            "log_filename_prefix": "example_bm25",
            "console_enabled": True,
            "file_level": "DEBUG",
        }
    }
    lightmem = LightMemory.from_config(config)
    return lightmem


def main():
    """
    Run a minimal demonstration of LightMemory add/retrieve workflow (BM25 version).
    """
    print("========== LightMemory BM25 Example ==========")
    lightmem = load_lightmem_bm25(EXAMPLE_COLLECTION)

    # ============ Add Example Memory ============
    messages = [
        {"role": "user", "content": "The capital of France is Paris."},
        {"role": "assistant", "content": "Correct, Paris is the capital city of France."}
    ]
    result = lightmem.add_memory(messages=messages, force_segment=True, force_extract=True)
    print("Memory added:", result)

    # ============ Retrieve Example Memory ============
    query = "What is the capital of France?"
    results = lightmem.retrieve(query, limit=5)
    print("Query:", query)
    print("Retrieved Results:", results)
    print("===============================================")


# ============ Entry Point ============
if __name__ == "__main__":
    main()
