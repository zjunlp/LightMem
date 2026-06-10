"""Procedural skill retriever via Distill edge traversal

Implements the procedural retrieval formula from the paper:
    V_proc_t = ∪_{v_epi ∈ V_epi_t} {v_proc | (v_epi, v_proc) ∈ E_distill}

Given the retrieved episodic nodes, collects related procedural skill nodes via DistillEdge.
"""
from typing import List

from ..graph.memory_graph import MemoryGraph
from ..graph.nodes import ProceduralNode


class ProceduralRetriever:
    """Procedural skill retriever

    Does not retrieve based on a query directly; instead, starts from the
    already-retrieved episodic nodes and collects related procedural skill
    nodes via E_distill edges.

    DistillEdge direction: episodic -> procedural
    (source_id is the episodic node id, target_id is the skill node id)
    """

    def __init__(self, graph: MemoryGraph):
        self.graph = graph

    def retrieve_for_episodes(
        self, episode_ids: List[str]
    ) -> List[ProceduralNode]:
        """Get related skill nodes from the retrieved episodic nodes via distill edges.

        Iterates over all DistillEdges; if the source_id is in episode_ids,
        the ProceduralNode pointed to by target_id is collected into the result.

        Uses graph.get_skills_for_episodes() to avoid duplicate implementation.

        Args:
            episode_ids: List of retrieved episodic node ids

        Returns:
            Deduplicated list of related ProceduralNode
        """
        if not episode_ids:
            return []

        return self.graph.get_skills_for_episodes(episode_ids)