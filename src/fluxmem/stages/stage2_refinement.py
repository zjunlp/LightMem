"""Stage II: Feedback-Driven Connectivity Refinement

Upon receiving execution feedback f_t, perform structural edits:
- Connection-Level: (i) Link Expansion (ii) Link Pruning
- Unit-Level: (iii) Content Reshaping for Granularity Alignment

Loop until execution succeeds or T rounds of refinement are reached
"""
import copy
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..graph.memory_graph import MemoryGraph, Subgraph
from ..graph.nodes import (
    BaseNode,
    SemanticNode,
    EpisodicNode,
    ProceduralNode,
    NodeType,
)
from ..graph.edges import StepLinkEdge, BaseEdge
from ..interfaces.llm import BaseLLM
from ..interfaces.embedder import BaseEmbedder
from ..retrieval.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)


class StageII:
    """Stage II: Feedback-Driven Connectivity Refinement

    Upon receiving execution feedback f_t, perform structural edits:
    - Connection-Level: (i) Link Expansion (ii) Link Pruning
    - Unit-Level: (iii) Content Reshaping for Granularity Alignment

    Loop until execution succeeds or T rounds of refinement are reached
    """

    def __init__(
        self,
        graph: MemoryGraph,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        semantic_retriever: SemanticRetriever,
        max_rounds: int = 5,
        expand_top_k: int = 3,
        prune_similarity_threshold: float = 0.3,
    ):
        self.graph = graph
        self.llm = llm
        self.embedder = embedder
        self.semantic_retriever = semantic_retriever
        self.max_rounds = max_rounds
        self.expand_top_k = expand_top_k
        self.prune_similarity_threshold = prune_similarity_threshold

    async def execute(
        self,
        subgraph: Subgraph,
        task_query: str,
        observation: str,
        feedback: str,
        step_index: int = 0,
    ) -> Tuple[Subgraph, str, bool]:
        """Run a single refinement round.

        Args:
            subgraph: Current subgraph G_t
            task_query: Task description
            observation: Current observation
            feedback: Execution feedback f_t
            step_index: Time step index

        Returns:
            (refined_subgraph, refined_context, success): refined subgraph, context and whether successful
        """
        # 1. Use the LLM to attribute the failure: attribute_failure(context, feedback)
        context_str = subgraph.to_context_string(task_query, observation)
        attribution = await self.llm.attribute_failure(context_str, feedback)

        logger.info(
            "Stage II attribution: type=%s, action=%s, details=%s",
            attribution.get("type"),
            attribution.get("action"),
            attribution.get("details", "")[:100],
        )

        action = attribution.get("action", "reshape")
        details = attribution.get("details", "")

        # 2. Apply edits according to the attribution type
        if action == "expand":
            refined_subgraph = await self._expand_links(
                subgraph, {"details": details}, observation, step_index
            )
        elif action == "prune":
            refined_subgraph = await self._prune_links(
                subgraph, {"details": details}
            )
        elif action == "reshape":
            refined_subgraph = await self._reshape_content(
                subgraph, {"details": details}, task_query, observation
            )
        else:
            # Unknown action; default to reshape
            logger.warning("Unknown action '%s', falling back to reshape", action)
            refined_subgraph = await self._reshape_content(
                subgraph, {"details": details}, task_query, observation
            )

        # 3. Re-serialize the context S'_t(q)
        refined_context = refined_subgraph.to_context_string(task_query, observation)

        return refined_subgraph, refined_context, False

    async def run_refinement_loop(
        self,
        subgraph: Subgraph,
        task_query: str,
        observation: str,
        execute_fn: Callable,
        step_index: int = 0,
    ) -> Tuple[Subgraph, str]:
        """Full refinement loop.

        Args:
            subgraph: Initial subgraph produced by Stage I
            task_query: Task description
            observation: Observation
            execute_fn: async callable; takes context_string and returns (result, feedback, success)
            step_index: Time step

        Returns:
            Final refined (subgraph, context_string)
        """
        current_subgraph = subgraph
        current_context = subgraph.to_context_string(task_query, observation)

        for round_idx in range(self.max_rounds):
            logger.info("Refinement round %d/%d", round_idx + 1, self.max_rounds)

            # 1. Run execute_fn(context) -> (result, feedback, success)
            try:
                result, feedback, success = await execute_fn(current_context)
            except Exception as e:
                logger.error("Execute function failed: %s", e)
                feedback = f"Execution error: {e}"
                success = False

            # 2. If success=True, terminate
            if success:
                logger.info("Refinement succeeded at round %d", round_idx + 1)
                return current_subgraph, current_context

            # 3. Otherwise call self.execute(..., feedback) to refine
            try:
                refined_subgraph, refined_context, _ = await self.execute(
                    subgraph=current_subgraph,
                    task_query=task_query,
                    observation=observation,
                    feedback=feedback,
                    step_index=step_index,
                )
                current_subgraph = refined_subgraph
                current_context = refined_context
            except Exception as e:
                logger.error("Refinement round %d failed: %s", round_idx + 1, e)
                # Refinement itself errored; keep current state and continue
                continue

        # Reached the maximum number of rounds without success
        logger.warning(
            "Refinement loop exhausted %d rounds without success", self.max_rounds
        )
        return current_subgraph, current_context

    async def _expand_links(
        self,
        subgraph: Subgraph,
        details: dict,
        observation: str,
        step_index: int,
    ) -> Subgraph:
        """Link Expansion: find inactive nodes and create connections.

        From V\\V_t (semantic nodes in the graph that are not in the current subgraph),
        find inactive nodes that are semantically close to the observation and add new edges.
        """
        # Get the set of node ids currently in the subgraph
        active_ids = set(subgraph.nodes.keys())

        # Retrieve semantic nodes related to the observation from the graph
        try:
            candidates = await self.semantic_retriever.retrieve(
                query=observation, top_k=self.expand_top_k * 3
            )
        except Exception as e:
            logger.warning("Expand link retrieval failed: %s", e)
            return subgraph

        # Filter nodes not yet active in the subgraph
        new_nodes: List[SemanticNode] = []
        for node in candidates:
            if node.id not in active_ids:
                new_nodes.append(node)

        if not new_nodes:
            logger.info("No new nodes found for expansion")
            return subgraph

        # Take the top-k new nodes and add them to the subgraph
        new_nodes = new_nodes[: self.expand_top_k]

        # Make a shallow copy of the subgraph to avoid direct mutation
        new_subgraph_nodes = dict(subgraph.nodes)
        new_subgraph_edges = dict(subgraph.edges)

        for node in new_nodes:
            # Add the new node
            new_subgraph_nodes[node.id] = node

            # Create a StepLinkEdge connecting the new node
            edge = StepLinkEdge(
                source_id=node.id,
                target_id=node.id,
                step_index=step_index,
                metadata={"retrieval_type": "expansion"},
            )
            new_subgraph_edges[edge.id] = edge

            # Sync to the global graph
            if node.id not in self.graph.semantic_nodes:
                self.graph.add_node(node)
            if edge.id not in self.graph.step_edges:
                self.graph.step_edges[edge.id] = edge

        logger.info("Expanded subgraph with %d new semantic nodes", len(new_nodes))

        return Subgraph(nodes=new_subgraph_nodes, edges=new_subgraph_edges)

    async def _prune_links(
        self, subgraph: Subgraph, details: dict
    ) -> Subgraph:
        """Link Pruning: identify and remove distracting edges.

        Identify the noise edges E_noise within the subgraph:
        - StepLinkEdges with low relevance to the current task
        - Use the embedder to compute the similarity between the node content and the noise
          description in the feedback; edges below the threshold are marked as noise edges and removed.
        """
        details_str = details.get("details", "")
        new_subgraph_nodes = dict(subgraph.nodes)
        new_subgraph_edges = dict(subgraph.edges)

        # Collect ids of edges to remove
        edge_ids_to_prune: List[str] = []

        # Iterate over StepLinkEdges in the subgraph and assess relevance to the feedback
        for eid, edge in list(subgraph.edges.items()):
            if not isinstance(edge, StepLinkEdge):
                continue

            # Self-loop edges (source_id == target_id) represent node activation;
            # determine whether the node content is related to the noise indicated in the feedback
            if edge.source_id == edge.target_id:
                node = subgraph.nodes.get(edge.source_id)
                if node is None:
                    continue

                # Use LLM verify to judge whether this node is a distractor
                try:
                    # Construct a "reverse query": is this node related to the noise description in the feedback?
                    # If the node content is highly related to the noise description, it is a noise source and should be pruned
                    noise_relevance = await self.llm.verify(
                        claim=details_str,
                        evidence=node.content if hasattr(node, 'content') and node.content else "",
                    )
                    # If this node is highly related to the noise description, mark it as a distractor and remove it
                    if noise_relevance > 0.7:
                        edge_ids_to_prune.append(eid)
                        logger.info(
                            "Pruning noisy node %s (relevance=%.2f)",
                            node.id, noise_relevance,
                        )
                except Exception as e:
                    logger.warning("LLM verify failed for pruning: %s", e)
                    continue

            else:
                # Non self-loop StepLinkEdge (e.g., epi -> proc)
                # Evaluate the relevance between the edge target node and the feedback
                target_node = subgraph.nodes.get(edge.target_id)
                if target_node is None:
                    continue
                try:
                    content = (
                        target_node.content
                        if hasattr(target_node, "content") and target_node.content
                        else ""
                    )
                    if not content:
                        content = (
                            target_node.skill_text
                            if hasattr(target_node, "skill_text")
                            else ""
                        )
                    noise_relevance = await self.llm.verify(
                        claim=details_str,
                        evidence=content,
                    )
                    if noise_relevance > 0.7:
                        edge_ids_to_prune.append(eid)
                except Exception:
                    continue

        if not edge_ids_to_prune:
            logger.info("No noisy edges identified for pruning")
            return subgraph

        # Remove distracting edges
        pruned_node_ids = set()
        for eid in edge_ids_to_prune:
            edge = new_subgraph_edges.pop(eid, None)
            if edge is not None:
                # Also remove the step edge from the global graph
                self.graph.step_edges.pop(eid, None)
                # Check whether the nodes have other connecting edges; if not, also remove them from the subgraph
                pruned_node_ids.add(edge.source_id)
                pruned_node_ids.add(edge.target_id)

        # Check whether nodes whose edges were removed are still connected to other edges
        for nid in pruned_node_ids:
            still_connected = any(
                (e.source_id == nid or e.target_id == nid)
                for e in new_subgraph_edges.values()
            )
            if not still_connected and nid in new_subgraph_nodes:
                # Remove orphaned node from the subgraph
                del new_subgraph_nodes[nid]
                logger.info("Removed isolated node %s after pruning", nid)

        logger.info("Pruned %d noisy edges", len(edge_ids_to_prune))

        return Subgraph(nodes=new_subgraph_nodes, edges=new_subgraph_edges)

    async def _reshape_content(
        self,
        subgraph: Subgraph,
        details: dict,
        task_query: str,
        observation: str,
    ) -> Subgraph:
        """Content Reshaping: adjust node granularity.

        Use BaseLLM.reshape_content() to create new content and replace the node.
        Based on the details returned by attribute_failure, decide whether to refine to fine
        or summarize to coarse; default is medium.
        """
        details_str = details.get("details", "")

        # Infer target granularity from details
        target_granularity = self._infer_granularity(details_str)

        new_subgraph_nodes = dict(subgraph.nodes)
        new_subgraph_edges = dict(subgraph.edges)

        # Identify nodes that need reshaping
        # Prefer reshaping semantic nodes (whose content most likely needs granularity adjustment)
        nodes_to_reshape = subgraph.semantic_nodes
        if not nodes_to_reshape:
            # If there are no semantic nodes, try episodic nodes
            nodes_to_reshape = subgraph.episodic_nodes

        if not nodes_to_reshape:
            logger.info("No nodes available for reshaping")
            return subgraph

        # Build the context used for reshaping
        reshape_context = f"Task: {task_query}\nObservation: {observation}\nFeedback: {details_str}"

        for node in nodes_to_reshape:
            # Get node content
            if isinstance(node, SemanticNode):
                original_content = node.content
            elif isinstance(node, EpisodicNode):
                # For episodic nodes, concatenate task_description and trajectory
                traj_str = "; ".join(
                    f"obs={o}, act={a}" for o, a in node.trajectory
                )
                original_content = f"Task: {node.task_description}. Trajectory: {traj_str}"
            elif isinstance(node, ProceduralNode):
                original_content = node.skill_text
            else:
                continue

            if not original_content:
                continue

            try:
                # Call LLM reshape_content
                reshaped_content = await self.llm.reshape_content(
                    node_content=original_content,
                    target_granularity=target_granularity,
                    context=reshape_context,
                )
            except Exception as e:
                logger.warning("Reshape failed for node %s: %s", node.id, e)
                continue

            if not reshaped_content:
                continue

            # Create a new node to replace the old one
            new_node = self._create_reshaped_node(node, reshaped_content)

            # Replace it in the subgraph
            old_id = node.id
            new_subgraph_nodes.pop(old_id, None)
            new_subgraph_nodes[new_node.id] = new_node

            # Redirect edges in the subgraph
            new_edges = {}
            for eid, edge in new_subgraph_edges.items():
                # Need to create a new edge object (since source_id/target_id on the dataclass are mutable)
                if edge.source_id == old_id:
                    edge.source_id = new_node.id
                if edge.target_id == old_id:
                    edge.target_id = new_node.id
                new_edges[eid] = edge
            new_subgraph_edges = new_edges

            # Replace it in the global graph
            self.graph.replace_node(old_id, new_node)

            logger.info(
                "Reshaped node %s -> %s (granularity=%s)",
                old_id[:8],
                new_node.id[:8],
                target_granularity,
            )

        return Subgraph(nodes=new_subgraph_nodes, edges=new_subgraph_edges)

    # ==================== Utility methods ====================

    @staticmethod
    def _infer_granularity(details_str: str) -> str:
        """Infer the target granularity from the failure attribution details.

        If details mention too much information / distracting / redundant -> coarse (summarize)
        If details mention insufficient information / missing / not enough detail -> fine (refine)
        Otherwise -> medium
        """
        details_lower = details_str.lower()

        coarse_keywords = [
            "too much", "irrelevant", "noisy", "redundant",
            "overwhelming", "excessive", "distracting",
            "过多", "冗余", "干扰", "不相关",
        ]
        fine_keywords = [
            "missing", "insufficient", "lack", "incomplete",
            "not enough", "need more detail", "absent",
            "缺失", "不足", "不够", "遗漏",
        ]

        for kw in coarse_keywords:
            if kw in details_lower:
                return "coarse"

        for kw in fine_keywords:
            if kw in details_lower:
                return "fine"

        return "medium"

    @staticmethod
    def _create_reshaped_node(original_node: BaseNode, new_content: str) -> BaseNode:
        """Create a new node with reshaped content based on the original node.

        Preserves the original node's type and key attributes but uses a new id and new content.
        """
        if isinstance(original_node, SemanticNode):
            return SemanticNode(
                source=original_node.source,
                chunk_index=original_node.chunk_index,
                content=new_content,
                embedding=original_node.embedding,  # Embedding is not updated yet; can be rebuilt asynchronously later
                metadata={
                    **original_node.metadata,
                    "reshaped_from": original_node.id,
                },
            )
        elif isinstance(original_node, EpisodicNode):
            return EpisodicNode(
                task_id=original_node.task_id,
                task_description=original_node.task_description,
                trajectory=original_node.trajectory,
                success=original_node.success,
                content=new_content,
                embedding=original_node.embedding,
                metadata={
                    **original_node.metadata,
                    "reshaped_from": original_node.id,
                },
            )
        elif isinstance(original_node, ProceduralNode):
            return ProceduralNode(
                skill_text=new_content,
                version=original_node.version + 1,
                version_history=original_node.version_history + [original_node.skill_text],
                source_episode_ids=original_node.source_episode_ids,
                pems_score=original_node.pems_score,
                content=new_content,
                embedding=original_node.embedding,
                metadata={
                    **original_node.metadata,
                    "reshaped_from": original_node.id,
                },
            )
        else:
            # Generic BaseNode
            return BaseNode(
                content=new_content,
                embedding=original_node.embedding,
                metadata={
                    **original_node.metadata,
                    "reshaped_from": original_node.id,
                },
            )
