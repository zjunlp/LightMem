from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import uuid
from enum import Enum

import numpy as np


class NodeType(Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


@dataclass
class BaseNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = field(init=False)
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class SemanticNode(BaseNode):
    """Semantic knowledge node - stores fact chunks and their embeddings"""
    node_type: NodeType = field(default=NodeType.SEMANTIC, init=False)
    source: str = ""  # Source document identifier
    chunk_index: int = 0  # Position of the chunk within the document


@dataclass
class EpisodicNode(BaseNode):
    """Episodic experience node - records task trajectory tau={(o_t, a_t)}"""
    node_type: NodeType = field(default=NodeType.EPISODIC, init=False)
    task_id: str = ""
    task_description: str = ""
    trajectory: List[Tuple[str, str]] = field(default_factory=list)  # [(observation, action), ...]
    success: bool = False


@dataclass
class ProceduralNode(BaseNode):
    """Procedural skill node - encapsulates a distilled reasoning template"""
    node_type: NodeType = field(default=NodeType.PROCEDURAL, init=False)
    skill_text: str = ""  # Skill description text
    version: int = 0
    version_history: List[str] = field(default_factory=list)  # Historical skill text versions
    source_episode_ids: List[str] = field(default_factory=list)  # Source episodic node ids
    pems_score: float = 0.0  # Current PEMS score
