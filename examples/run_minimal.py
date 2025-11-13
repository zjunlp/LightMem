"""
A minimal example to run LightMemory with only context-based retrieval (BM25).

This script demonstrates how to:
1. Configure LightMemory to use only the BM25 context retriever.
2. Manually create and save memory entries to a JSON file to seed the retriever.
3. Instantiate LightMemory with the minimal configuration.
4. Add a new memory entry using `add_memory`.
5. Retrieve memories using the `retrieve` method.

This setup avoids the need for heavy dependencies like torch, transformers, and OpenAI.
"""
import os
import sys
import json

# Ensure we can import the src package directly without installation
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from lightmem.memory.lightmem import LightMemory
from lightmem.memory.utils import MemoryEntry, save_memory_entries
from lightmem.configs.base import BaseMemoryConfigs
from lightmem.configs.memory_manager.base import MemoryManagerConfig
from lightmem.configs.retriever.contextretriever.base import ContextRetrieverConfig

# 1. Define a minimal configuration
minimal_config = BaseMemoryConfigs(
    memory_manager=MemoryManagerConfig(model_name="openai", configs={}),
    index_strategy="context",
    retrieve_strategy="context",
    metadata_generate=False,
    text_summary=False,
    topic_segment=False, # Disable topic segmentation
    update="online", # Use online update to immediately save memories
    context_retriever=ContextRetrieverConfig(model_name="BM25", configs={"corpus_path": "minimal_memory.json"})
)

# 2. Create a dummy memory file for the BM25 retriever
if not os.path.exists("minimal_memory.json"):
    print("Creating dummy memory file...")
    initial_memories = [
        MemoryEntry(memory="The user's favorite color is blue."),
        MemoryEntry(memory="The user lives in New York.")
    ]
    save_memory_entries(initial_memories, "minimal_memory.json")

# 3. Instantiate LightMemory
print("Initializing LightMemory...")
memory = LightMemory(config=minimal_config)

# 4. Add a new memory
print("Adding a new memory...")
new_message = {
    "role": "user",
    "content": "The user's favorite food is pizza.",
    "time_stamp": "2024-01-01 (Mon) 12:00"
}
# Since metadata_generate is False, add_memory won't do much, so we'll add manually for this example
new_memory_entry = MemoryEntry(memory=new_message["content"])
save_memory_entries([new_memory_entry], "minimal_memory.json")

# Re-initialize to load the new memory
memory = LightMemory(config=minimal_config)


# 5. Retrieve memories
print("Retrieving memories...")
query = "What is the user's favorite food?"
results = memory.retrieve(query=query, limit=1)

print("\n--- Retrieval Results ---")
print(results)
print("-----------------------")

# Clean up the dummy file
os.remove("minimal_memory.json")