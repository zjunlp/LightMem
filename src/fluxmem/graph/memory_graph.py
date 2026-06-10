from typing import Dict, List, Optional, Tuple
import numpy as np

from .nodes import (
    BaseNode, SemanticNode, EpisodicNode, ProceduralNode, NodeType
)
from .edges import (
    BaseEdge, GroundEdge, DistillEdge, StepLinkEdge, EdgeType
)


class Subgraph:
    """Local subgraph G_t(q) = (V_t, E_t)"""

    def __init__(self, nodes: Dict[str, BaseNode], edges: Dict[str, BaseEdge]):
        self.nodes = nodes
        self.edges = edges

    @property
    def semantic_nodes(self) -> List[SemanticNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.SEMANTIC]

    @property
    def episodic_nodes(self) -> List[EpisodicNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.EPISODIC]

    @property
    def procedural_nodes(self) -> List[ProceduralNode]:
        return [n for n in self.nodes.values() if n.node_type == NodeType.PROCEDURAL]

    def to_context_string(self, task_query: str, observation: str) -> str:
        """Format the subgraph contents as structured text for the LLM to understand.

        Use XML tags to organize the three layers of memory:
        - <semantic_memory>: relevant factual knowledge
        - <episodic_memory>: relevant historical experience
        - <procedural_memory>: relevant skill templates
        """
        parts: List[str] = []

        parts.append("<context>")
        parts.append(f"  <task_query>{task_query}</task_query>")
        parts.append(f"  <current_observation>{observation}</current_observation>")

        # --- Semantic memory layer ---
        sem_nodes = self.semantic_nodes
        if sem_nodes:
            parts.append("  <semantic_memory>")
            for i, node in enumerate(sem_nodes, 1):
                parts.append(f'    <fact id="{node.id}" source="{node.source}" chunk="{node.chunk_index}">')
                parts.append(f"      {node.content}")
                parts.append("    </fact>")
            parts.append("  </semantic_memory>")
        else:
            parts.append("  <semantic_memory />")

        # --- Episodic memory layer ---
        epi_nodes = self.episodic_nodes
        if epi_nodes:
            parts.append("  <episodic_memory>")
            for node in epi_nodes:
                success_tag = "success" if node.success else "failure"
                parts.append(f'    <episode id="{node.id}" task="{node.task_description}" outcome="{success_tag}">')
                if node.trajectory:
                    for step_idx, (obs, act) in enumerate(node.trajectory):
                        parts.append(f'      <step index="{step_idx}">')
                        parts.append(f"        <observation>{obs}</observation>")
                        parts.append(f"        <action>{act}</action>")
                        parts.append("      </step>")
                parts.append("    </episode>")
            parts.append("  </episodic_memory>")
        else:
            parts.append("  <episodic_memory />")

        # --- Procedural memory layer ---
        proc_nodes = self.procedural_nodes
        if proc_nodes:
            parts.append("  <procedural_memory>")
            for node in proc_nodes:
                parts.append(
                    f'    <skill id="{node.id}" '
                    f'version="{node.version}" '
                    f'pems_score="{node.pems_score:.2f}">'
                )
                parts.append(f"      {node.skill_text}")
                parts.append("    </skill>")
            parts.append("  </procedural_memory>")
        else:
            parts.append("  <procedural_memory />")

        parts.append("</context>")
        return "\n".join(parts)


class MemoryGraph:
    """Heterogeneous memory graph G=(V,E)"""

    def __init__(self):
        # Three layers of node storage
        self.semantic_nodes: Dict[str, SemanticNode] = {}
        self.episodic_nodes: Dict[str, EpisodicNode] = {}
        self.procedural_nodes: Dict[str, ProceduralNode] = {}
        # Edge storage
        self.ground_edges: Dict[str, GroundEdge] = {}
        self.distill_edges: Dict[str, DistillEdge] = {}
        self.step_edges: Dict[str, StepLinkEdge] = {}

    # ========== Node operations ==========

    def add_node(self, node: BaseNode) -> str:
        """Add a node to the dictionary of its type and return the node id"""
        if node.node_type == NodeType.SEMANTIC:
            self.semantic_nodes[node.id] = node  # type: ignore[assignment]
        elif node.node_type == NodeType.EPISODIC:
            self.episodic_nodes[node.id] = node  # type: ignore[assignment]
        elif node.node_type == NodeType.PROCEDURAL:
            self.procedural_nodes[node.id] = node  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")
        return node.id

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its associated edges; return whether deletion succeeded"""
        node = self.get_node(node_id)
        if node is None:
            return False

        # Remove associated edges
        related_edge_ids = []
        for edge_dict in (self.ground_edges, self.distill_edges, self.step_edges):
            for eid, edge in list(edge_dict.items()):
                if edge.source_id == node_id or edge.target_id == node_id:
                    related_edge_ids.append(eid)

        for eid in related_edge_ids:
            self.remove_edge(eid)

        # Remove from the node dictionary
        if node.node_type == NodeType.SEMANTIC:
            self.semantic_nodes.pop(node_id, None)
        elif node.node_type == NodeType.EPISODIC:
            self.episodic_nodes.pop(node_id, None)
        elif node.node_type == NodeType.PROCEDURAL:
            self.procedural_nodes.pop(node_id, None)

        return True

    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Look up a node by id, searching all three layers"""""
        if node_id in self.semantic_nodes:
            return self.semantic_nodes[node_id]
        if node_id in self.episodic_nodes:
            return self.episodic_nodes[node_id]
        if node_id in self.procedural_nodes:
            return self.procedural_nodes[node_id]
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> List[BaseNode]:
        """Get all nodes of the specified type"""""
        if node_type == NodeType.SEMANTIC:
            return list(self.semantic_nodes.values())
        elif node_type == NodeType.EPISODIC:
            return list(self.episodic_nodes.values())
        elif node_type == NodeType.PROCEDURAL:
            return list(self.procedural_nodes.values())
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    # ========== Edge operations ==========

    def add_edge(self, edge: BaseEdge) -> str:
        """Add an edge to the dictionary of its type and return the edge id"""""
        if edge.edge_type == EdgeType.GROUND:
            self.ground_edges[edge.id] = edge  # type: ignore[assignment]
        elif edge.edge_type == EdgeType.DISTILL:
            self.distill_edges[edge.id] = edge  # type: ignore[assignment]
        elif edge.edge_type == EdgeType.STEP_LINK:
            self.step_edges[edge.id] = edge  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown edge type: {edge.edge_type}")
        return edge.id

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge; return whether deletion succeeded"""""
        if edge_id in self.ground_edges:
            del self.ground_edges[edge_id]
            return True
        if edge_id in self.distill_edges:
            del self.distill_edges[edge_id]
            return True
        if edge_id in self.step_edges:
            del self.step_edges[edge_id]
            return True
        return False

    def get_edges_from(self, node_id: str) -> List[BaseEdge]:
        """Get all edges originating from the specified node"""""
        result: List[BaseEdge] = []
        for edge_dict in (self.ground_edges, self.distill_edges, self.step_edges):
            for edge in edge_dict.values():
                if edge.source_id == node_id:
                    result.append(edge)
        return result

    def get_edges_to(self, node_id: str) -> List[BaseEdge]:
        """Get all edges pointing to the specified node"""""
        result: List[BaseEdge] = []
        for edge_dict in (self.ground_edges, self.distill_edges, self.step_edges):
            for edge in edge_dict.values():
                if edge.target_id == node_id:
                    result.append(edge)
        return result

    # ========== Subgraph extraction ==========

    def extract_subgraph(self, node_ids: List[str]) -> Subgraph:
        """Extract a local subgraph from a list of node ids, including the nodes and edges between them"""""
        nodes: Dict[str, BaseNode] = {}
        for nid in node_ids:
            node = self.get_node(nid)
            if node is not None:
                nodes[nid] = node

        node_id_set = set(node_ids)
        edges: Dict[str, BaseEdge] = {}
        for edge_dict in (self.ground_edges, self.distill_edges, self.step_edges):
            for eid, edge in edge_dict.items():
                if edge.source_id in node_id_set and edge.target_id in node_id_set:
                    edges[eid] = edge

        return Subgraph(nodes=nodes, edges=edges)

    # ========== Context serialization ==========

    def serialize_context(self, task_query: str, observation: str,
                          sem_nodes: List[SemanticNode],
                          epi_nodes: List[EpisodicNode],
                          proc_nodes: List[ProceduralNode]) -> str:
        """Format the contents of the specified nodes as structured text for LLM consumption.

        Uses XML tags to organize the three memory layers, consistent with Subgraph.to_context_string.
        """
        parts: List[str] = []

        parts.append("<context>")
        parts.append(f"  <task_query>{task_query}</task_query>")
        parts.append(f"  <current_observation>{observation}</current_observation>")

        # --- Semantic memory layer ---
        if sem_nodes:
            parts.append("  <semantic_memory>")
            for node in sem_nodes:
                parts.append(f'    <fact id="{node.id}" source="{node.source}" chunk="{node.chunk_index}">')
                parts.append(f"      {node.content}")
                parts.append("    </fact>")
            parts.append("  </semantic_memory>")
        else:
            parts.append("  <semantic_memory />")

        # --- Episodic memory layer ---
        if epi_nodes:
            parts.append("  <episodic_memory>")
            for node in epi_nodes:
                success_tag = "success" if node.success else "failure"
                parts.append(f'    <episode id="{node.id}" task="{node.task_description}" outcome="{success_tag}">')
                if node.trajectory:
                    for step_idx, (obs, act) in enumerate(node.trajectory):
                        parts.append(f'      <step index="{step_idx}">')
                        parts.append(f"        <observation>{obs}</observation>")
                        parts.append(f"        <action>{act}</action>")
                        parts.append("      </step>")
                parts.append("    </episode>")
            parts.append("  </episodic_memory>")
        else:
            parts.append("  <episodic_memory />")

        # --- Procedural memory layer ---
        if proc_nodes:
            parts.append("  <procedural_memory>")
            for node in proc_nodes:
                parts.append(
                    f'    <skill id="{node.id}" '
                    f'version="{node.version}" '
                    f'pems_score="{node.pems_score:.2f}">'
                )
                parts.append(f"      {node.skill_text}")
                parts.append("    </skill>")
            parts.append("  </procedural_memory>")
        else:
            parts.append("  <procedural_memory />")

        parts.append("</context>")
        return "\n".join(parts)

    # ========== Topology editing (used in Stage II) ==========

    def expand_link(self, anchor_id: str, new_node_id: str, step: int) -> str:
        """Expand a step link: create a StepLinkEdge between the anchor node and the new node.

        Used in Stage II to insert a new step into an existing episodic trajectory.
        Returns the id of the newly created edge.
        """
        edge = StepLinkEdge(
            source_id=anchor_id,
            target_id=new_node_id,
            step_index=step,
        )
        self.step_edges[edge.id] = edge
        return edge.id

    def prune_links(self, edge_ids: List[str]) -> int:
        """Prune edges in batch; return the number of edges successfully removed"""""
        count = 0
        for eid in edge_ids:
            if self.remove_edge(eid):
                count += 1
        return count

    def replace_node(self, old_id: str, new_node: BaseNode) -> bool:
        """Replace a node: replace the old node with a new one, preserving all edges that reference the old node (redirect them to the new node).

        Returns whether the replacement succeeded.
        """
        old_node = self.get_node(old_id)
        if old_node is None:
            return False

        # First remove the old node (without deleting associated edges)
        if old_node.node_type == NodeType.SEMANTIC:
            self.semantic_nodes.pop(old_id, None)
        elif old_node.node_type == NodeType.EPISODIC:
            self.episodic_nodes.pop(old_id, None)
        elif old_node.node_type == NodeType.PROCEDURAL:
            self.procedural_nodes.pop(old_id, None)

        # Redirect edges: update any edge referencing old_id to use new_node.id
        for edge_dict in (self.ground_edges, self.distill_edges, self.step_edges):
            for edge in edge_dict.values():
                if edge.source_id == old_id:
                    edge.source_id = new_node.id
                if edge.target_id == old_id:
                    edge.target_id = new_node.id

        # Add the new node
        self.add_node(new_node)
        return True

    # ========== Query helpers ==========

    def get_skills_for_episodes(self, episode_ids: List[str]) -> List[ProceduralNode]:
        """Get procedural skill nodes connected to the specified episodic nodes via DistillEdge"""""
        epi_id_set = set(episode_ids)
        proc_ids: set = set()

        for edge in self.distill_edges.values():
            if edge.source_id in epi_id_set:
                proc_ids.add(edge.target_id)

        result: List[ProceduralNode] = []
        for pid in proc_ids:
            node = self.procedural_nodes.get(pid)
            if node is not None:
                result.append(node)
        return result

    def get_all_semantic_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Get embeddings for all semantic nodes.

        Returns (list of ids, embedding matrix); returns an empty list and an empty array when there are no nodes.
        """
        ids: List[str] = []
        embeddings: List[np.ndarray] = []
        for node in self.semantic_nodes.values():
            if node.embedding is not None:
                ids.append(node.id)
                embeddings.append(node.embedding)

        if not embeddings:
            return ids, np.array([])
        return ids, np.stack(embeddings)

    def get_all_episodic_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Get embeddings for all episodic nodes.

        Returns (list of ids, embedding matrix); returns an empty list and an empty array when there are no nodes.
        """
        ids: List[str] = []
        embeddings: List[np.ndarray] = []
        for node in self.episodic_nodes.values():
            if node.embedding is not None:
                ids.append(node.id)
                embeddings.append(node.embedding)

        if not embeddings:
            return ids, np.array([])
        return ids, np.stack(embeddings)
