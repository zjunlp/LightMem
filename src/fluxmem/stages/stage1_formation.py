"""Stage I: Initial Connection Formation

At each time step t, build initial connections:
1. Semantic Connection Retrieval: query V_sem to obtain top-k semantic nodes
2. Episodic Connection Retrieval: query V_epi to obtain top-k episodic nodes
3. Procedural Connection Inheritance: obtain related skills via E_distill

Output: initial subgraph G_t and serialized context S_t(q)
"""
import logging
from typing import Dict, List, Tuple

import numpy as np

from ..graph.memory_graph import MemoryGraph, Subgraph
from ..graph.nodes import SemanticNode, EpisodicNode, ProceduralNode, NodeType
from ..graph.edges import StepLinkEdge
from ..retrieval.semantic_retriever import SemanticRetriever
from ..retrieval.episodic_retriever import EpisodicRetriever
from ..retrieval.procedural_retriever import ProceduralRetriever
from ..interfaces.llm import BaseLLM
from ..interfaces.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class StageI:
    """Stage I: Initial Connection Formation

    At each time step t, build initial connections:
    1. Semantic Connection Retrieval: query V_sem to obtain top-k semantic nodes
    2. Episodic Connection Retrieval: query V_epi to obtain top-k episodic nodes
    3. Procedural Connection Inheritance: obtain related skills via E_distill

    Output: initial subgraph G_t and serialized context S_t(q)
    """

    def __init__(
        self,
        graph: MemoryGraph,
        semantic_retriever: SemanticRetriever,
        episodic_retriever: EpisodicRetriever,
        procedural_retriever: ProceduralRetriever,
        embedder: BaseEmbedder,
        llm: BaseLLM,
        top_k_semantic: int = 5,
        top_k_episodic: int = 3,
    ):
        self.graph = graph
        self.semantic_retriever = semantic_retriever
        self.episodic_retriever = episodic_retriever
        self.procedural_retriever = procedural_retriever
        self.embedder = embedder
        self.llm = llm
        self.top_k_semantic = top_k_semantic
        self.top_k_episodic = top_k_episodic

    async def execute(
        self, task_query: str, observation: str, step_index: int = 0
    ) -> Tuple[Subgraph, str]:
        """Execute Stage I and return (subgraph, context_string).

        Args:
            task_query: Current task description
            observation: Observation o_t for the current step
            step_index: Index of the current time step

        Returns:
            (Subgraph, str): initial subgraph G_t and the serialized context S_t(q)
        """
        # 1. Semantic Connection Retrieval
        # Score(v, o_t) = cosine + BM25 + LLM_ver, take top-k
        semantic_nodes = await self._retrieve_semantic(observation)

        # 2. Episodic Connection Retrieval
        # V_epi_t = TopK cos(u, o_t)
        episodic_nodes = await self._retrieve_episodic(observation)

        # 3. Procedural Connection Inheritance
        # V_proc_t = ∪ {v_proc | (v_epi, v_proc) ∈ E_distill}
        procedural_nodes = self._retrieve_procedural(episodic_nodes)

        # 4. Build the subgraph and create StepLinkEdge connections
        subgraph = self._build_subgraph(
            semantic_nodes, episodic_nodes, procedural_nodes, step_index
        )

        # 5. Serialize as context string S_t(q) = Concat(q, Obs_t, V_sem_t, V_epi_t, V_proc_t)
        context_string = subgraph.to_context_string(task_query, observation)

        logger.info(
            "Stage I formed subgraph: %d semantic, %d episodic, %d procedural nodes",
            len(semantic_nodes),
            len(episodic_nodes),
            len(procedural_nodes),
        )

        return subgraph, context_string

    # ==================== Internal methods ====================

    async def _retrieve_semantic(self, observation: str) -> List[SemanticNode]:
        """Semantic connection retrieval: use hybrid scoring to obtain top-k semantic nodes.

        Score(v, o_t) = dense_weight * cosine_sim + bm25_weight * BM25 + llm_weight * LLM_ver
        """
        try:
            nodes = await self.semantic_retriever.retrieve(
                query=observation, top_k=self.top_k_semantic
            )
            return nodes
        except Exception as e:
            logger.warning("Semantic retrieval failed: %s", e)
            return []

    async def _retrieve_episodic(self, observation: str) -> List[EpisodicNode]:
        """Episodic connection retrieval: V_epi_t = TopK_{u ∈ V_epi} cos(u, o_t)"""
        try:
            nodes = await self.episodic_retriever.retrieve(
                query=observation, top_k=self.top_k_episodic
            )
            return nodes
        except Exception as e:
            logger.warning("Episodic retrieval failed: %s", e)
            return []

    def _retrieve_procedural(
        self, episodic_nodes: List[EpisodicNode]
    ) -> List[ProceduralNode]:
        """Procedural connection inheritance: obtain related skills via E_distill.

        V_proc_t = ∪_{v_epi ∈ V_epi_t} {v_proc | (v_epi, v_proc) ∈ E_distill}
        """
        if not episodic_nodes:
            return []
        episode_ids = [node.id for node in episodic_nodes]
        return self.procedural_retriever.retrieve_for_episodes(episode_ids)

    def _build_subgraph(
        self,
        semantic_nodes: List[SemanticNode],
        episodic_nodes: List[EpisodicNode],
        procedural_nodes: List[ProceduralNode],
        step_index: int,
    ) -> Subgraph:
        """Build the initial subgraph and create StepLinkEdge connections.

        Collect the three categories of nodes into the subgraph and create a
        StepLinkEdge from the "query anchor" to each node, annotated with the
        current time step step_index.
        """
        # Collect all nodes
        nodes: Dict[str, object] = {}
        for node in semantic_nodes:
            nodes[node.id] = node
        for node in episodic_nodes:
            nodes[node.id] = node
        for node in procedural_nodes:
            nodes[node.id] = node

        # Build StepLinkEdge: create step connections for each retrieved node
        edges: Dict[str, StepLinkEdge] = {}

        # StepLink between semantic nodes -> episodic nodes (reflects activation at this time step)
        for sem_node in semantic_nodes:
            edge = StepLinkEdge(
                source_id=sem_node.id,
                target_id=sem_node.id,  # Self-loop indicates the node is activated at step_index
                step_index=step_index,
                metadata={"retrieval_type": "semantic"},
            )
            edges[edge.id] = edge
            # Also register on the graph for later editing in Stage II
            if edge.id not in self.graph.step_edges:
                self.graph.step_edges[edge.id] = edge

        for epi_node in episodic_nodes:
            edge = StepLinkEdge(
                source_id=epi_node.id,
                target_id=epi_node.id,
                step_index=step_index,
                metadata={"retrieval_type": "episodic"},
            )
            edges[edge.id] = edge
            if edge.id not in self.graph.step_edges:
                self.graph.step_edges[edge.id] = edge

        # StepLink from episodic nodes to related procedural skill nodes
        for epi_node in episodic_nodes:
            for proc_node in procedural_nodes:
                # Check if a distill edge connects them
                has_distill = any(
                    e.source_id == epi_node.id and e.target_id == proc_node.id
                    for e in self.graph.distill_edges.values()
                )
                if has_distill:
                    edge = StepLinkEdge(
                        source_id=epi_node.id,
                        target_id=proc_node.id,
                        step_index=step_index,
                        metadata={"retrieval_type": "procedural_inherit"},
                    )
                    edges[edge.id] = edge
                    if edge.id not in self.graph.step_edges:
                        self.graph.step_edges[edge.id] = edge

        for proc_node in procedural_nodes:
            edge = StepLinkEdge(
                source_id=proc_node.id,
                target_id=proc_node.id,
                step_index=step_index,
                metadata={"retrieval_type": "procedural"},
            )
            edges[edge.id] = edge
            if edge.id not in self.graph.step_edges:
                self.graph.step_edges[edge.id] = edge

        # Also include existing internal edges of the graph (ground/distill) within the subgraph
        node_id_set = set(nodes.keys())
        for eid, edge in self.graph.ground_edges.items():
            if edge.source_id in node_id_set and edge.target_id in node_id_set:
                edges[eid] = edge
        for eid, edge in self.graph.distill_edges.items():
            if edge.source_id in node_id_set and edge.target_id in node_id_set:
                edges[eid] = edge

        return Subgraph(nodes=nodes, edges=edges)
