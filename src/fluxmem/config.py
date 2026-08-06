from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FluxMemConfig:
    """FluxMem framework configuration"""

    # Retrieval parameters
    top_k_semantic: int = 5          # Top-k for semantic retrieval
    top_k_episodic: int = 3          # Top-k for episodic retrieval

    # Stage II parameters
    max_refinement_rounds: int = 5   # T: maximum number of refinement rounds

    # Stage III parameters
    num_clusters: Optional[int] = None  # Number of clusters; None for automatic
    max_consolidation_rounds: int = 5   # Maximum number of consolidation iterations
    pems_threshold: float = 0.01        # epsilon: PEMS convergence threshold

    # Retrieval weights
    dense_weight: float = 1.0
    bm25_weight: float = 0.5
    llm_weight: float = 0.3

    # Embedding dimension
    embedding_dimension: int = 1536

    # LLM configuration
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7