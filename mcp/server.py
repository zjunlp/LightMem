import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from fastmcp import FastMCP
except ImportError:
    print("fastmcp module is not installed. Please install it to proceed.")
    sys.exit(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from lightmem.memory.lightmem import LightMemory
except ImportError:
    print("LightMemory module is not found. Please ensure LightMem is properly installed.")
    sys.exit(1)


# -----------------------------
# Init LightMemory
# -----------------------------

_lightmem_instance: Optional[LightMemory] = None

# the default config path is `example.json` in the same directory as this script
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example.json')

def get_lightmem_instance() -> LightMemory:
    """
    Delays the initialization of the LightMemory instance and ensures that all tools share the same instance.
    """
    global _lightmem_instance

    if _lightmem_instance is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Configuration file does not exist: {CONFIG_PATH}")

        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        _lightmem_instance = LightMemory.from_config(config)

    return _lightmem_instance


# -----------------------------
# MCP Initialization
# -----------------------------

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

mcp = FastMCP("LightMem")

@mcp.tool()
def get_timestamp() -> Dict[str, Any]:
    """
    Get the current time and return it in the specified format (YYYY-MM-DDTHH:MM:SS.sss).

    Returns:
        A dictionary containing the operation result
    """
    timestamp = datetime.now().isoformat(timespec="milliseconds")
    if timestamp:
        return {
            "status": STATUS_SUCCESS,
            "message": timestamp,
        }
    else:
        return {
            "status": STATUS_ERROR,
            "message": "Failed to get the current timestamp.",
        }

@mcp.tool()
def add_memory(user_input: str, assistant_reply: str, timestamp: Optional[str] = None, force_segment: bool = False, force_extract: bool = False) -> Dict[str, Any]:
    """
    Add new memory (user input and assistant reply pair) to the LightMem.

    Args:
        user_input: User role's input or question
        assistant_reply: Assistant role's response or reply
        timestamp: Timestamp (optional, format: "YYYY-MM-DDTHH:MM:SS.sss")
        force_segment: Whether to force segmentation, regardless of buffer conditions
        force_extract: Whether to force memory extraction, regardless of thresholds

    Returns:
        A dictionary containing the operation result
    """
    lightmem_instance = get_lightmem_instance()

    if lightmem_instance is None:
        return {
            "status": STATUS_ERROR,
            "message": "LightMem is not initialized. Please check the configuration file."
        }

    try:
        if not user_input or not assistant_reply:
            return {
                "status": STATUS_ERROR,
                "message": "Both `user_input` and `assistant_reply` are required."
            }

        timestamp = timestamp or datetime.now().isoformat(timespec="milliseconds")

        full_message = [
            {
                "role": "user",
                "content": user_input,
                "time_stamp": timestamp
            },
            {
                "role": "assistant",
                "content": assistant_reply,
                "time_stamp": timestamp
            }
        ]

        added_result = lightmem_instance.add_memory(
            messages=full_message,
            force_segment=force_segment,
            force_extract=force_extract
        )

        if (
            "triggered" in added_result and 
            "emitted_messages" in added_result
        ):
            return {
                "status": STATUS_SUCCESS,
                "message": "Topic segmentation is disabled; memory pipeline returned early.",
                "details": {
                    "triggered": added_result.get("triggered"),
                    "cut_index": added_result.get("cut_index"),
                    "boundaries": added_result.get("boundaries"),
                    "emitted_messages": added_result.get("emitted_messages"),
                    "carryover_size": added_result.get("carryover_size"),
                }
            }

        if (
            "add_input_prompt" in added_result and 
            "add_output_prompt" in added_result
        ):
            return {
                "status": STATUS_SUCCESS,
                "message": "Memory has been successfully added to LightMem.",
                "details": {
                    "add_input_prompt": added_result.get("add_input_prompt", []),
                    "add_output_prompt": added_result.get("add_output_prompt", []),
                    "api_call_nums": added_result.get("api_call_nums", 0),
                }
            }

        return {
            "status": STATUS_ERROR,
            "message": "LightMem `add_memory` returned an unexpected structure.",
            "details": {
                "raw_return": added_result
            }
        }

    except Exception as e:
        return {
            "status": STATUS_ERROR,
            "message": f"Error adding memory: {str(e)}"
        }

@mcp.tool()
def offline_update(top_k: int = 20, keep_top_n: int = 10, score_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Update all memory entries by using LightMem's update strategy.

    Args:
        top_k: Number of nearest neighbors to consider for each entry
        keep_top_n: Number of top entries to keep in update_queue
        score_threshold: Minimum similarity score for considering update candidates

    Returns:
        A dictionary containing the operation result
    """
    lightmem_instance = get_lightmem_instance()

    if lightmem_instance is None:
        return {
            "status": STATUS_ERROR,
            "message": "LightMem is not initialized. Please check the configuration file."
        }

    try:
        lightmem_instance.construct_update_queue_all_entries(
            top_k=top_k,
            keep_top_n=keep_top_n
        )
        lightmem_instance.offline_update_all_entries(
            score_threshold=score_threshold
        )

        return {
            "status": STATUS_SUCCESS,
            "message": "Offline update completed successfully."
        }

    except Exception as e:
        return {
            "status": STATUS_ERROR,
            "message": f"Error during offline update: {str(e)}"
        }

@mcp.tool()
def retrieve_memory(query: str, limit: int = 10, filters: Optional[Any] = {}) -> Dict[str, Any]:
    """
    Retrieve relevant memory entries from LightMem based on a query.

    Args:
        query: The natural language query string to search for relevant memories
        limit: Number of similar results to return (top-k for vector retrieval)
        filters: Optional filters to narrow down the search, usually metadata filters supported by the vector database.

    Returns:
        A dictionary containing the operation result
    """
    lightmem_instance = get_lightmem_instance()
    
    if lightmem_instance is None:
        return {
            "status": STATUS_ERROR,
            "message": "LightMem is not initialized. Please check the configuration file."
        }

    if filters == {}:
        filters = None

    if not query:
        return {
            "status": STATUS_ERROR,
            "message": "query parameter is required"
        }

    try:
        related_memories = lightmem_instance.retrieve(
            query=query,
            limit=limit,
            filters=filters
        )
        if isinstance(related_memories, str):
            related_memories_list = [item for item in related_memories.split("\n") if item.strip()]
        elif isinstance(related_memories, list):
            related_memories_list = related_memories
        else:
            related_memories_list = [str(related_memories)]

        return {
            "status": STATUS_SUCCESS,
            "message": f"LightMem has retrieved {len(related_memories_list)} relevant memories.",
            "details": related_memories_list
        }

    except Exception as e:
        return {
            "status": STATUS_ERROR,
            "message": f"Error retrieving memory: {str(e)}"
        }

@mcp.tool()
def show_lightmem_instance() -> Dict[str, Any]:
    """
    Show the current LightMem instance's status.

    Returns:
        A dictionary containing the operation result
    """
    lightmem_instance = get_lightmem_instance()

    if lightmem_instance is None:
        return {
            "status": STATUS_ERROR,
            "message": "LightMem is not initialized. Please check the configuration file."
        }

    try:
        show = {}
        show["lightmem"] = lightmem_instance
        show["config"] = lightmem_instance.config
        show["compressor"] = lightmem_instance.compressor
        show["segmenter"] = lightmem_instance.segmenter
        show["manager"] = lightmem_instance.manager
        show["text_embedder"] = lightmem_instance.text_embedder
        show["retrieve_strategy"] = lightmem_instance.retrieve_strategy
        if lightmem_instance.retrieve_strategy in ["context", "hybrid"]:
            show["context_retriever"] = lightmem_instance.context_retriever
        if lightmem_instance.retrieve_strategy in ["embedding", "hybrid"]:
            show["embedding_retriever"] = lightmem_instance.embedding_retriever
        show["logger"] = lightmem_instance.logger

        readable_show = json.dumps({k: str(v) for k, v in show.items()}, indent=2, ensure_ascii=False)

        return {
            "status": STATUS_SUCCESS,
            "message": "LightMem instance details retrieved successfully.",
            "details": readable_show
        }

    except Exception as e:
        return {
            "status": STATUS_ERROR,
            "message": f"Error retrieving configuration: {str(e)}"
        }


# -----------------------------
# Main Function
# -----------------------------

def main():
    global CONFIG_PATH

    parser = argparse.ArgumentParser(description="an MCP server for LightMem")
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
    )
    args = parser.parse_args()

    CONFIG_PATH = args.config

    try:
        print("Using config:", CONFIG_PATH)
        print("Starting MCP server...")
        mcp.run(single_thread=True) # Single thread

    except KeyboardInterrupt:
        print("Server interrupted by user", file=sys.stderr)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
for example, to run the MCP server, use the following commands:

(lightmem) xxx/LightMem$ npx @modelcontextprotocol/inspector python mcp/server.py

(lightmem) xxx/LightMem$ fastmcp run mcp/server.py:mcp --transport http --port 8000
"""
