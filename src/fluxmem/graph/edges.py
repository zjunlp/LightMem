from dataclasses import dataclass, field
from typing import Dict, Any
import uuid
from enum import Enum


class EdgeType(Enum):
    GROUND = "ground"        # E_ground: semantic -> episodic
    DISTILL = "distill"      # E_distill: episodic -> procedural
    STEP_LINK = "step_link"  # Step-level temporary connection edge


@dataclass
class BaseEdge:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""  # Source node id
    target_id: str = ""  # Target node id
    weight: float = 1.0
    created_at: float = field(default_factory=lambda: __import__('time').time())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundEdge(BaseEdge):
    """E_ground ⊆ V_sem × V_epi: facts provide evidential support for episodic steps"""
    edge_type: EdgeType = field(default=EdgeType.GROUND, init=False)


@dataclass
class DistillEdge(BaseEdge):
    """E_distill ⊆ V_epi × V_proc: skills distilled from experience"""
    edge_type: EdgeType = field(default=EdgeType.DISTILL, init=False)


@dataclass
class StepLinkEdge(BaseEdge):
    """Step-level temporary connection, created in Stage I and editable in Stage II"""
    edge_type: EdgeType = field(default=EdgeType.STEP_LINK, init=False)
    step_index: int = 0  # Associated time step
