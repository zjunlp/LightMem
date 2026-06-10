"""FluxMem Agent main controller class

Orchestrates the three-stage memory evolution pipeline:
- Stage I: Initial Connection Formation (online, executed each step)
- Stage II: Feedback-Driven Connectivity Refinement (online, executed each step)
- Stage III: Long-Term Connection Consolidation (offline, executed periodically)
"""
import logging
import uuid
from typing import Optional, List, Callable, Awaitable, Dict, Any, Tuple

import numpy as np

from .config import FluxMemConfig
from .graph.memory_graph import MemoryGraph, Subgraph
from .graph.nodes import SemanticNode, EpisodicNode, ProceduralNode, NodeType
from .graph.edges import GroundEdge
from .interfaces.llm import BaseLLM
from .interfaces.embedder import BaseEmbedder
from .interfaces.vectorstore import BaseVectorStore, FAISSVectorStore
from .retrieval.semantic_retriever import SemanticRetriever
from .retrieval.episodic_retriever import EpisodicRetriever
from .retrieval.procedural_retriever import ProceduralRetriever
from .stages.stage1_formation import StageI
from .stages.stage2_refinement import StageII
from .stages.stage3_consolidation import StageIII
from .metrics.pems import PEMSCalculator

logger = logging.getLogger(__name__)


class FluxMemAgent:
    """FluxMem Agent main controller class

    Orchestrates the three-stage memory evolution pipeline:
    - Stage I: Initial Connection Formation (online, executed each step)
    - Stage II: Feedback-Driven Connectivity Refinement (online, executed each step)
    - Stage III: Long-Term Connection Consolidation (offline, executed periodically)
    """

    def __init__(
        self,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        config: Optional[FluxMemConfig] = None,
        semantic_vectorstore: Optional[BaseVectorStore] = None,
        episodic_vectorstore: Optional[BaseVectorStore] = None,
    ):
        """Initialize the FluxMemAgent.

        Args:
            llm: LLM instance
            embedder: Embedding model instance
            config: Configuration; uses defaults when None
            semantic_vectorstore: Semantic vector store; uses FAISSVectorStore when None
            episodic_vectorstore: Episodic vector store; uses FAISSVectorStore when None
        """
        self.config = config or FluxMemConfig()
        self.llm = llm
        self.embedder = embedder
        self.graph = MemoryGraph()

        # Initialize vector stores (one for semantic and one for episodic)
        if semantic_vectorstore is not None:
            self.semantic_vs = semantic_vectorstore
        else:
            self.semantic_vs = FAISSVectorStore(
                dimension=self.config.embedding_dimension
            )

        if episodic_vectorstore is not None:
            self.episodic_vs = episodic_vectorstore
        else:
            self.episodic_vs = FAISSVectorStore(
                dimension=self.config.embedding_dimension
            )

        # Initialize the PEMS calculator
        self.pems_calculator = PEMSCalculator(
            convergence_threshold=self.config.pems_threshold
        )

        # Initialize the three retrievers
        self.semantic_retriever = SemanticRetriever(
            graph=self.graph,
            embedder=self.embedder,
            llm=self.llm,
            vectorstore=self.semantic_vs,
            dense_weight=self.config.dense_weight,
            bm25_weight=self.config.bm25_weight,
            llm_weight=self.config.llm_weight,
        )

        self.episodic_retriever = EpisodicRetriever(
            graph=self.graph,
            embedder=self.embedder,
            vectorstore=self.episodic_vs,
        )

        self.procedural_retriever = ProceduralRetriever(
            graph=self.graph,
        )

        # Initialize the three stages
        self.stage1 = StageI(
            graph=self.graph,
            semantic_retriever=self.semantic_retriever,
            episodic_retriever=self.episodic_retriever,
            procedural_retriever=self.procedural_retriever,
            embedder=self.embedder,
            llm=self.llm,
            top_k_semantic=self.config.top_k_semantic,
            top_k_episodic=self.config.top_k_episodic,
        )

        self.stage2 = StageII(
            graph=self.graph,
            llm=self.llm,
            embedder=self.embedder,
            semantic_retriever=self.semantic_retriever,
            max_rounds=self.config.max_refinement_rounds,
        )

        self.stage3 = StageIII(
            graph=self.graph,
            llm=self.llm,
            embedder=self.embedder,
            pems_calculator=self.pems_calculator,
            num_clusters=self.config.num_clusters,
            max_consolidation_rounds=self.config.max_consolidation_rounds,
            convergence_threshold=self.config.pems_threshold,
        )

        # Task history records
        self._task_history: List[Dict[str, Any]] = []

    # === Knowledge management API ===

    async def add_knowledge(
        self, text: str, source: str = "", chunk_size: int = 512
    ) -> List[str]:
        """Add semantic knowledge into the memory graph (auto-chunking and embedding).

        Splits long text by chunk_size, computes an embedding for each chunk,
        creates SemanticNode instances and adds them to the graph and vector index.

        Args:
            text: Knowledge text to add
            source: Source document identifier
            chunk_size: Chunk size in characters

        Returns:
            List of created SemanticNode ids
        """
        if not text:
            return []

        # 1. Chunking: split text by chunk_size
        chunks = self._split_text(text, chunk_size)
        if not chunks:
            return []

        # 2. Compute embeddings in batch
        embeddings = await self.embedder.embed_batch(chunks)

        # 3. Create SemanticNode instances and add them to the graph
        node_ids: List[str] = []
        for idx, (chunk_content, chunk_embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            node = SemanticNode(
                content=chunk_content,
                source=source,
                chunk_index=idx,
                embedding=chunk_embedding,
            )
            self.graph.add_node(node)
            node_ids.append(node.id)

        # 4. Add embeddings to the semantic vector store
        if embeddings.shape[0] > 0:
            self.semantic_vs.add(node_ids, embeddings)

        # 5. Update BM25 index
        self.semantic_retriever.build_index()

        logger.info(
            "Added %d semantic chunks from source '%s'",
            len(node_ids),
            source,
        )
        return node_ids

    async def add_knowledge_nodes(
        self, nodes: List[SemanticNode]
    ) -> List[str]:
        """Add already-built semantic nodes directly.

        For nodes lacking an embedding, compute the embedding automatically and
        add the nodes to the graph and the vector index.

        Args:
            nodes: List of pre-built SemanticNode instances

        Returns:
            List of added node ids
        """
        if not nodes:
            return []

        # 1. Compute embeddings for nodes that lack one
        nodes_without_emb = [n for n in nodes if n.embedding is None]
        if nodes_without_emb:
            contents = [n.content for n in nodes_without_emb]
            embeddings = await self.embedder.embed_batch(contents)
            for node, emb in zip(nodes_without_emb, embeddings):
                node.embedding = emb

        # 2. Add to the graph and the vector index
        node_ids: List[str] = []
        all_ids: List[str] = []
        all_embeddings: List[np.ndarray] = []

        for node in nodes:
            self.graph.add_node(node)
            node_ids.append(node.id)
            if node.embedding is not None:
                all_ids.append(node.id)
                all_embeddings.append(node.embedding)

        # 3. Add to the vector store in batch
        if all_ids and all_embeddings:
            emb_matrix = np.stack(all_embeddings)
            self.semantic_vs.add(all_ids, emb_matrix)

        # 4. Update BM25 index
        self.semantic_retriever.build_index()

        logger.info("Added %d semantic nodes directly", len(node_ids))
        return node_ids

    # === Task execution API ===

    async def run_task(
        self,
        task_query: str,
        execute_fn: Callable[[str], Awaitable[Tuple[str, str, bool]]],
        observations: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a complete task.

        Runs the Stage I -> Stage II loop for each step, and submits the trajectory
        as an EpisodicNode after the task ends.

        Args:
            task_query: Task description
            execute_fn: Execution function that accepts a context_string and
                        returns (result, feedback, success)
            observations: Predefined observation sequence. If None, uses the
                          execute_fn result as the observation for the next step

        Returns:
            Dict containing: {
                'success': bool,
                'trajectory': [(observation, action), ...],
                'steps': int,
                'episode_id': str  # id of the generated episodic node
            }
        """
        trajectory: List[Tuple[str, str]] = []
        overall_success = False
        step_index = 0

        # Determine the observation sequence
        if observations is not None and len(observations) > 0:
            # Use the predefined observation sequence
            for step_index, observation in enumerate(observations):
                step_result = await self.run_step(
                    task_query=task_query,
                    observation=observation,
                    execute_fn=execute_fn,
                    step_index=step_index,
                )

                # Record the trajectory
                trajectory.append((observation, step_result.get("result", "")))

                if step_result.get("success", False):
                    overall_success = True
                    break  # Stop on success
        else:
            # Dynamic observations: use task description as the initial observation
            observation = task_query
            max_steps = self.config.max_refinement_rounds + 1  # Prevent infinite loop

            for step_index in range(max_steps):
                step_result = await self.run_step(
                    task_query=task_query,
                    observation=observation,
                    execute_fn=execute_fn,
                    step_index=step_index,
                )

                result_text = step_result.get("result", "")
                trajectory.append((observation, result_text))

                if step_result.get("success", False):
                    overall_success = True
                    break

                # The next observation is the current execution result
                observation = result_text if result_text else observation

        # After task completion: create the EpisodicNode
        episode_node = EpisodicNode(
            task_id=str(uuid.uuid4()),
            task_description=task_query,
            trajectory=trajectory,
            success=overall_success,
            content=f"Task: {task_query}. Steps: {len(trajectory)}. "
                    f"Outcome: {'success' if overall_success else 'failure'}",
        )

        # Compute the embedding for the episodic node
        episode_text = episode_node.content
        episode_node.embedding = await self.embedder.embed_text(episode_text)

        # Add to the memory graph
        self.graph.add_node(episode_node)

        # Add to the episodic vector store
        self.episodic_vs.add(
            [episode_node.id],
            episode_node.embedding.reshape(1, -1),
        )

        # Create GroundEdges between the episodic node and related semantic nodes
        # Connect semantic knowledge involved in the trajectory to the episode
        if trajectory:
            # Retrieve semantic nodes related to the task
            relevant_semantic = await self.semantic_retriever.retrieve(
                query=task_query,
                top_k=self.config.top_k_semantic,
            )
            for sem_node in relevant_semantic:
                edge = GroundEdge(
                    source_id=sem_node.id,
                    target_id=episode_node.id,
                )
                self.graph.add_edge(edge)

        # Record task history
        task_record = {
            "task_query": task_query,
            "success": overall_success,
            "steps": len(trajectory),
            "episode_id": episode_node.id,
            "trajectory": trajectory,
        }
        self._task_history.append(task_record)

        logger.info(
            "Task completed: success=%s, steps=%d, episode_id=%s",
            overall_success,
            len(trajectory),
            episode_node.id[:8],
        )

        return {
            "success": overall_success,
            "trajectory": trajectory,
            "steps": len(trajectory),
            "episode_id": episode_node.id,
        }

    async def run_step(
        self,
        task_query: str,
        observation: str,
        execute_fn: Callable[[str], Awaitable[Tuple[str, str, bool]]],
        step_index: int = 0,
    ) -> Dict[str, Any]:
        """Execute a single step (Stage I + Stage II).

        Args:
            task_query: Task description
            observation: Current observation
            execute_fn: Execution function
            step_index: Step index

        Returns:
            Dict: {'result': str, 'success': bool, 'context': str, 'subgraph': Subgraph}
        """
        # 1. Stage I: build initial connections
        subgraph, context_string = await self.stage1.execute(
            task_query=task_query,
            observation=observation,
            step_index=step_index,
        )

        # 2. Try executing once first to decide whether refinement is needed
        try:
            result, feedback, success = await execute_fn(context_string)
        except Exception as e:
            logger.warning("Initial execution failed: %s", e)
            result = ""
            feedback = f"Execution error: {e}"
            success = False

        if success:
            # Return immediately on success
            return {
                "result": result,
                "success": True,
                "context": context_string,
                "subgraph": subgraph,
            }

        # 3. Stage II: refinement loop (enter refinement on failure)
        refined_subgraph, refined_context = await self.stage2.run_refinement_loop(
            subgraph=subgraph,
            task_query=task_query,
            observation=observation,
            execute_fn=execute_fn,
            step_index=step_index,
        )

        # After the refinement loop ends, execute again to obtain the final result
        try:
            final_result, final_feedback, final_success = await execute_fn(
                refined_context
            )
        except Exception as e:
            logger.warning("Final execution after refinement failed: %s", e)
            final_result = ""
            final_feedback = f"Execution error: {e}"
            final_success = False

        return {
            "result": final_result if final_result else result,
            "success": final_success,
            "context": refined_context,
            "subgraph": refined_subgraph,
        }

    # === Offline consolidation API ===

    async def consolidate(
        self, replay_fn: Optional[Callable] = None
    ) -> Dict:
        """Run Stage III offline consolidation.

        Args:
            replay_fn: Optional replay function used to validate skills

        Returns:
            Consolidation result dict containing:
            - clusters: clustering results
            - induced_skills: list of induced skill nodes
            - pems_history: PEMS iteration history
            - converged: whether the process has converged
        """
        result = await self.stage3.execute(replay_fn=replay_fn)

        # Rebuild vector indices after consolidation (new skill node embeddings may exist)
        self.rebuild_indices()

        logger.info(
            "Consolidation completed: converged=%s, skills=%d",
            result.get("converged", False),
            len(result.get("induced_skills", [])),
        )

        return result

    # === Index management ===

    def rebuild_indices(self) -> None:
        """Rebuild all vector indices.

        Re-collects all node embeddings from the memory graph and rebuilds the
        semantic and episodic vector store indices, as well as the BM25 index.
        """
        # Rebuild the semantic vector index
        self.semantic_vs = FAISSVectorStore(
            dimension=self.config.embedding_dimension
        )
        ids, embeddings = self.graph.get_all_semantic_embeddings()
        if ids and embeddings.size > 0:
            self.semantic_vs.add(ids, embeddings)

        # Update the vectorstore reference of the semantic retriever
        self.semantic_retriever.vectorstore = self.semantic_vs

        # Rebuild the episodic vector index
        self.episodic_vs = FAISSVectorStore(
            dimension=self.config.embedding_dimension
        )
        ids, embeddings = self.graph.get_all_episodic_embeddings()
        if ids and embeddings.size > 0:
            self.episodic_vs.add(ids, embeddings)

        # Update the vectorstore reference of the episodic retriever
        self.episodic_retriever.vectorstore = self.episodic_vs

        # Rebuild the BM25 index
        self.semantic_retriever.build_index()

        logger.info("Rebuilt all vector indices")

    # === Status queries ===

    @property
    def num_semantic_nodes(self) -> int:
        """Number of semantic nodes"""
        return len(self.graph.semantic_nodes)

    @property
    def num_episodic_nodes(self) -> int:
        """Number of episodic nodes"""
        return len(self.graph.episodic_nodes)

    @property
    def num_procedural_nodes(self) -> int:
        """Number of procedural nodes"""
        return len(self.graph.procedural_nodes)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return framework statistics"""
        return {
            "num_semantic_nodes": self.num_semantic_nodes,
            "num_episodic_nodes": self.num_episodic_nodes,
            "num_procedural_nodes": self.num_procedural_nodes,
            "num_ground_edges": len(self.graph.ground_edges),
            "num_distill_edges": len(self.graph.distill_edges),
            "num_step_edges": len(self.graph.step_edges),
            "num_tasks": len(self._task_history),
            "semantic_vs_size": len(self.semantic_vs),
            "episodic_vs_size": len(self.episodic_vs),
        }

    # === Utility methods ===

    @staticmethod
    def _split_text(text: str, chunk_size: int) -> List[str]:
        """Split text by chunk_size, preferring breaks at sentence or paragraph boundaries.

        Args:
            text: Text to split
            chunk_size: Maximum number of characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # If not at the end of text, look for a good split point within the chunk_size range
            if end < len(text):
                # Search for sentence/paragraph boundaries between (start + chunk_size * 0.7) and end
                search_start = start + int(chunk_size * 0.7)
                search_end = end

                best_split = end  # Default to a hard split
                # Prefer splitting on paragraph boundaries
                for sep in ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";"]:
                    pos = text.rfind(sep, search_start, search_end)
                    if pos != -1:
                        best_split = pos + len(sep)
                        break

                end = best_split

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks