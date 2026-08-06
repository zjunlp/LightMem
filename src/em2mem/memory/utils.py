from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from PIL import Image


@dataclass
class MemorySearchOutput:
    """Output structure for memory reasoning decisions."""
    memory_type: str
    search_query: str


@dataclass
class ReasoningOutput:
    """Output structure for the reasoning agent's decision."""
    decision: str  # "search" or "answer"
    selected_memory: Optional[MemorySearchOutput] = None
    reason: Optional[str] = None


@dataclass
class RetrievedItem:
    """Represents a single retrieved item from any memory type."""
    memory_type: str  # "episodic", "semantic", or "visual"
    content: Union[str, List[Image.Image]]  # Text for episodic/semantic, images for visual
    query: str  # The search query used
    round_num: int  # Which retrieval round


@dataclass 
class QAResult:
    """Result of the full QA pipeline."""
    question: str
    answer: str
    retrieved_items: List[RetrievedItem]
    round_history: List[Dict[str, Any]]
    num_rounds: int


def transform_timestamp(ts_str: str) -> str:
    """Transform timestamp string to human-readable format."""
    if len(ts_str) < 7:
        return ts_str
    day = ts_str[0]
    time_str = ts_str[1:]
    hh = time_str[0:2]
    mm = time_str[2:4]
    ss = time_str[4:6]
    return f"DAY{day} {hh}:{mm}:{ss}"
