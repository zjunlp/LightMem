import os
import sys
import traceback

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from lightmem.memory.lightmem import LightMemory
    from lightmem.configs.base import BaseMemoryConfigs, MemoryManagerConfig
    from lightmem.configs.retriever.bm25 import BM25Config
    from lightmem.configs.logging.base import LoggingConfig
    from lightmem.configs.topic_segmenter.base import TopicSegmenterConfig 
except ImportError as e:
    print(f"--- Import failed! Error: {e} ---")
    print(f"--- Please ensure the 'src' directory is at: {src_path} ---")
    sys.exit()

# ============ Data Configuration ============
EXAMPLE_COLLECTION = "demo_collection_bm25"


def load_lightmem_bm25(collection_name):
    """
    Load a LightMemory instance configured for BM25 retrieval.
    """

    # 1. Define BM25 retriever configuration
    bm25_retriever_config = BM25Config(
        collection_name=collection_name,
        path=f"./bm25_data/{collection_name}"
    )

    # 2. Define Memory Manager configuration (requires OPENAI_API_KEY)
    manager_config = MemoryManagerConfig(
        model_name="openai",
        configs={
            "model": "gpt-3.5-turbo",
            "max_tokens": 2048,
        }
    )
    
    # 3. Define logging configuration
    logging_config = LoggingConfig(
        level="INFO",
        file_enabled=True,
        log_dir="logs",
        log_filename_prefix="example_bm25",
        console_enabled=True,
        file_level="DEBUG",
    )

    # 4. Define segmenter configuration (requires torch/llmlingua-2)
    segmenter_config = TopicSegmenterConfig(
        model_name="llmlingua-2" 
    )
    
    # 5. Assemble the final BaseMemoryConfigs
    config_object = BaseMemoryConfigs(
        pre_compress=False,
        topic_segment=True,  # Ensure add_memory does not exit prematurely
        topic_segmenter=segmenter_config,
        index_strategy="bm25",
        retrieve_strategy="bm25",
        bm25_retriever=bm25_retriever_config,
        memory_manager=manager_config,
        logging=logging_config
    )
    
    # 6. Initialize using the config object
    print("--- Initializing LightMemory (requires API Key and torch)... ---")
    lightmem = LightMemory.from_config(config_object.model_dump())
    print("--- LightMemory initialization successful ---")
    
    return lightmem


def main():
    """
    Run a minimal demonstration of LightMemory add/retrieve workflow (BM25 version).
    """
        
    print("\n========== LightMemory BM25 Example ==========")
    
    try:
        lightmem = load_lightmem_bm25(EXAMPLE_COLLECTION)

        # ============ Add Example Memory ============
        print("\n--- Attempting: lightmem.add_memory() ---")
        
        session_timestamp = "2025/11/12 (Wed) 19:30" 
        messages = [
            {"role": "user", "content": "The capital of France is Paris.", "time_stamp": session_timestamp},
            {"role": "assistant", "content": "Correct, Paris is the capital city of France.", "time_stamp": session_timestamp}
        ] 
        
        result = lightmem.add_memory(messages=messages, force_segment=True, force_extract=True)
        print("Memory added:", result)

        # ============ Retrieve Example Memory ============
        print("\n--- Attempting: lightmem.retrieve() ---")
        query = "What is the capital of France?"
        results = lightmem.retrieve(query, limit=5)
        print("Query:", query)
        print("Retrieved Results:", results)

    except Exception as e:
        print(f"\n--- Example run failed ---")
        print("This may be due to a missing OPENAI_API_KEY environment variable or missing 'torch' dependency.")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        
    print("\n===============================================")


# ============ Entry Point ============
if __name__ == "__main__":
    main()