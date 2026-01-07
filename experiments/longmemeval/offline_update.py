import os
from lightmem.memory.lightmem import LightMemory

def load_lightmem(collection_name):
    config = {
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": "gpt-4o-mini",
                "api_key": "",
                "max_tokens": 16000,
                "openai_base_url": ""
            }
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": 384,
                "path": f"/{collection_name}",
            }
        },
        "update": "offline",
    }
    lightmem = LightMemory.from_config(config)
    return lightmem

base_dir = ""

for collection_name in os.listdir(base_dir):
    collection_path = os.path.join(base_dir, collection_name)
    if not os.path.isdir(collection_path):
        continue  

    print(f"Processing collection: {collection_name}")

    try:
        lightmem = load_lightmem(collection_name)
        lightmem.construct_update_queue_all_entries()
        lightmem.offline_update_all_entries(score_threshold=0.8)
        print(f"Finished updating {collection_name}")
    except Exception as e:
        print(f"Error processing {collection_name}: {e}")
