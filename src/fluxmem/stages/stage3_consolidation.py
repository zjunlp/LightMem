"""Stage III: Long-Term Connection Consolidation (offline execution)

Pipeline:
1. Episodic Clustering: cluster episodic nodes by semantic similarity
2. Skill Induction: use the LLM to induce shared skills from each cluster
3. PEMS-Guided Iterative Consolidation: iteratively validate and refine skills until convergence
"""

from typing import List, Dict, Tuple, Optional, Callable, Awaitable
import logging

import numpy as np

from ..graph.memory_graph import MemoryGraph
from ..graph.nodes import EpisodicNode, ProceduralNode, NodeType
from ..graph.edges import DistillEdge
from ..interfaces.llm import BaseLLM
from ..interfaces.embedder import BaseEmbedder
from ..metrics.pems import PEMSCalculator

logger = logging.getLogger(__name__)

# Try to import sklearn; fall back when unavailable
try:
    from sklearn.cluster import KMeans as _SklearnKMeans

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class _SimpleKMeans:
    """Simple K-Means implementation, used as a fallback when sklearn is unavailable"""

    def __init__(self, n_clusters: int = 8, max_iter: int = 300, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_SimpleKMeans":
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if n_samples <= self.n_clusters:
            # Number of samples <= number of clusters: one cluster per sample
            self.labels_ = np.arange(n_samples)
            return self

        # Randomly choose initial centers
        indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centers = X[indices].copy()

        for _ in range(self.max_iter):
            # Assignment: each point is assigned to the nearest center
            distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centers
            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            # Convergence check
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        self.labels_ = labels
        return self


class StageIII:
    """Stage III: Long-Term Connection Consolidation (offline execution)

    1. Episodic Clustering: cluster episodic nodes by semantic similarity
    2. Skill Induction: use the LLM to induce shared skills from each cluster
    3. PEMS-Guided Iterative Consolidation: iteratively validate and refine skills
    """

    def __init__(
        self,
        graph: MemoryGraph,
        llm: BaseLLM,
        embedder: BaseEmbedder,
        pems_calculator: PEMSCalculator,
        num_clusters: Optional[int] = None,
        max_consolidation_rounds: int = 5,
        convergence_threshold: float = 0.01,
    ):
        self.graph = graph
        self.llm = llm
        self.embedder = embedder
        self.pems_calculator = pems_calculator
        self.num_clusters = num_clusters
        self.max_consolidation_rounds = max_consolidation_rounds
        self.convergence_threshold = convergence_threshold

    async def execute(
        self, replay_fn: Optional[Callable] = None
    ) -> Dict:
        """Run the full Stage III consolidation pipeline.

        Args:
            replay_fn: Optional replay function;
                       async callable(episode: EpisodicNode, skill: ProceduralNode) -> bool
                       used to re-execute an episode under the guidance of a skill and return success.
                       If None, estimate using the episode's own success field.

        Returns:
            Dict containing: {
                'clusters': clustering result,
                'induced_skills': list of induced skill nodes,
                'pems_history': PEMS iteration history,
                'converged': whether the process has converged
            }
        """
        # 1. Cluster episodic nodes
        clusters = self.cluster_episodes()
        logger.info(f"Stage III: clustered {sum(len(c) for c in clusters)} episodes "
                     f"into {len(clusters)} clusters")

        # 2. Induce a skill for each cluster
        induced_skills: List[ProceduralNode] = []
        for idx, cluster in enumerate(clusters):
            if not cluster:
                continue
            logger.info(f"Stage III: inducing skill from cluster {idx} "
                         f"({len(cluster)} episodes)")
            skill = await self.induce_skill(cluster)
            induced_skills.append(skill)

        # 3. Iteratively consolidate each skill
        all_pems_history: List[List[float]] = []
        all_converged = True

        for skill in induced_skills:
            source_episodes = [
                self.graph.episodic_nodes[eid]
                for eid in skill.source_episode_ids
                if eid in self.graph.episodic_nodes
            ]
            refined_skill, pems_hist = await self.iterative_consolidation(
                skill, source_episodes, replay_fn
            )
            all_pems_history.append(pems_hist)
            if pems_hist and abs(pems_hist[-1] - pems_hist[-2 if len(pems_hist) >= 2 else -1]) >= self.convergence_threshold:
                all_converged = False

        # Check whether all skills have converged
        converged = all_converged and len(induced_skills) > 0

        return {
            "clusters": clusters,
            "induced_skills": induced_skills,
            "pems_history": all_pems_history,
            "converged": converged,
        }

    def cluster_episodes(self) -> List[List[EpisodicNode]]:
        """Cluster V_epi based on cosine distance using K-Means or hierarchical clustering.

        Returns:
            List of clustering results; each element is a list of EpisodicNode in that cluster
        """
        episodes = list(self.graph.episodic_nodes.values())
        if not episodes:
            return []

        # Collect episodic nodes that have an embedding
        episodes_with_emb = [ep for ep in episodes if ep.embedding is not None]
        if not episodes_with_emb:
            # No embeddings available; place all nodes into a single cluster
            return [episodes] if episodes else []

        # If some nodes lack embeddings, handle the ones that have embeddings first
        episodes_without_emb = [ep for ep in episodes if ep.embedding is None]

        embeddings = np.stack([ep.embedding for ep in episodes_with_emb])

        # Determine the number of clusters
        n_clusters = self.num_clusters or self._determine_num_clusters(
            len(episodes_with_emb)
        )
        n_clusters = min(n_clusters, len(episodes_with_emb))

        if n_clusters <= 0:
            return [episodes]

        # Run clustering
        if _HAS_SKLEARN:
            kmeans = _SklearnKMeans(
                n_clusters=n_clusters, random_state=42, n_init=10
            )
            kmeans.fit(embeddings)
            labels = kmeans.labels_
        else:
            kmeans = _SimpleKMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(embeddings)
            labels = kmeans.labels_

        # Group by label
        clusters: List[List[EpisodicNode]] = [[] for _ in range(n_clusters)]
        for ep, label in zip(episodes_with_emb, labels):
            clusters[int(label)].append(ep)

        # Add nodes without embeddings to the largest cluster
        if episodes_without_emb:
            max_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
            clusters[max_idx].extend(episodes_without_emb)

        # Filter out empty clusters
        clusters = [c for c in clusters if c]
        return clusters

    async def induce_skill(self, cluster: List[EpisodicNode]) -> ProceduralNode:
        """Induce a skill from a cluster of episodes.

        Use the LLM to extract shared reasoning patterns/skills, and create a ProceduralNode and DistillEdge.

        Args:
            cluster: List of EpisodicNode in the same cluster

        Returns:
            The created ProceduralNode
        """
        # 1. Collect trajectory text from all episodes in the cluster
        trajectories: List[str] = []
        for ep in cluster:
            traj_parts: List[str] = []
            traj_parts.append(f"Task: {ep.task_description}")
            traj_parts.append(f"Outcome: {'success' if ep.success else 'failure'}")
            if ep.trajectory:
                for step_idx, (obs, act) in enumerate(ep.trajectory):
                    traj_parts.append(f"  Step {step_idx}: obs={obs} | action={act}")
            trajectories.append("\n".join(traj_parts))

        # 2. Call the LLM to induce common patterns
        skill_text = await self.llm.extract_skills(trajectories)

        # 3. Create the ProceduralNode
        source_ids = [ep.id for ep in cluster]
        skill_node = ProceduralNode(
            skill_text=skill_text,
            version=1,
            source_episode_ids=source_ids,
            pems_score=0.0,
        )

        # 4. Compute embedding
        skill_node.embedding = await self.embedder.embed_text(skill_text)

        # 5. Add the node to the graph
        self.graph.add_node(skill_node)

        # 6. Create a DistillEdge for each source episode (episodic -> procedural)
        for ep_id in source_ids:
            edge = DistillEdge(
                source_id=ep_id,
                target_id=skill_node.id,
                weight=1.0,
            )
            self.graph.add_edge(edge)

        return skill_node

    async def iterative_consolidation(
        self,
        skill: ProceduralNode,
        source_episodes: List[EpisodicNode],
        replay_fn: Optional[Callable] = None,
    ) -> Tuple[ProceduralNode, List[float]]:
        """PEMS-guided iterative consolidation.

        Runs a test-score-refine loop until delta_PEMS < epsilon or the maximum number of iterations is reached.

        Args:
            skill: Skill node to consolidate
            source_episodes: List of source episodic nodes
            replay_fn: Optional replay function

        Returns:
            (consolidated skill node, PEMS history)
        """
        # Reset the PEMS calculator for iterations on this skill
        pems_tracker = PEMSCalculator(
            convergence_threshold=self.convergence_threshold
        )
        pems_history: List[float] = []

        for round_k in range(1, self.max_consolidation_rounds + 1):
            logger.info(f"  Consolidation round {round_k} for skill "
                         f"'{skill.skill_text[:50]}...'")

            # 1. Evaluate: compute success rate eta
            if replay_fn is not None:
                success_count = 0
                for ep in source_episodes:
                    try:
                        result = await replay_fn(ep, skill)
                        if result:
                            success_count += 1
                    except Exception as e:
                        logger.warning(f"Replay failed for episode {ep.id}: {e}")
                success_rate = success_count / max(len(source_episodes), 1)
            else:
                success_rate = self._estimate_success_rate(skill, source_episodes)

            # 2. Compute PEMS score
            num_proc = len(self.graph.procedural_nodes)
            token_length = len(skill.skill_text.split())

            previous_embedding = None
            if skill.version > 1 and len(skill.version_history) > 0:
                # Recompute the embedding of the previous version
                previous_embedding = await self.embedder.embed_text(
                    skill.version_history[-1]
                )

            pems_score = pems_tracker.compute(
                success_rate=success_rate,
                num_proc_nodes=num_proc,
                skill_token_length=token_length,
                current_embedding=skill.embedding
                if skill.embedding is not None
                else np.zeros(self.embedder.dimension),
                previous_embedding=previous_embedding,
            )
            pems_history.append(pems_score)

            # Update the node's PEMS score
            skill.pems_score = pems_score

            # 3. Check convergence
            if pems_tracker.has_converged(pems_score):
                logger.info(f"  Skill converged at round {round_k}, "
                             f"PEMS={pems_score:.4f}")
                break

            # 4. If not converged and not the last round, refine the skill
            if round_k < self.max_consolidation_rounds:
                # Construct feedback: current PEMS and success rate info
                feedback = (
                    f"Current skill PEMS score: {pems_score:.4f}. "
                    f"Source episode success rate: {success_rate:.2%}. "
                    f"Consolidation round: {round_k}. "
                    f"The skill has not yet converged. "
                    f"Please improve the skill to better cover the source episodes."
                )

                # Save the current version into history
                skill.version_history.append(skill.skill_text)

                # Call the LLM to refine
                refined_text = await self.llm.refine_skill(skill.skill_text, feedback)

                # Update the skill node
                skill.skill_text = refined_text
                skill.version += 1

                # Recompute embedding
                skill.embedding = await self.embedder.embed_text(refined_text)

                logger.info(f"  Refined skill to version {skill.version}")
        else:
            logger.info(f"  Reached max consolidation rounds "
                         f"({self.max_consolidation_rounds}) without convergence")

        return skill, pems_history

    def _estimate_success_rate(
        self, skill: ProceduralNode, episodes: List[EpisodicNode]
    ) -> float:
        """Estimate success rate from the episodes' own success field when no replay_fn is given.

        Computes a weighted estimate using the success field of source episodes;
        the proportion of successful episodes is the estimated success rate.
        """
        if not episodes:
            return 0.0
        success_count = sum(1 for ep in episodes if ep.success)
        return success_count / len(episodes)

    def _determine_num_clusters(self, num_episodes: int) -> int:
        """Determine the number of clusters automatically.

        Uses a simple heuristic: sqrt(n/2), with a minimum of 1.
        """
        if num_episodes <= 0:
            return 1
        return max(1, int(np.sqrt(num_episodes / 2)))
