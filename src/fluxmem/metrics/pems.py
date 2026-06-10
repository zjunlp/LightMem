"""PEMS (Procedure Evolution Maturity Score) calculator

Formula: PEMS^(k) = eta^(k) / (|V_proc| * log(l^(k))) * (1 - delta(G^(k), G^(k-1)))

Where:
- eta^(k): average success rate of source episodes under the current skill
- |V_proc|: total number of procedural skill nodes (normalization factor)
- l^(k): token length of the skill text
- delta^(k): difference between the current and previous skill embeddings
"""

import math
import numpy as np
from typing import List, Optional


class PEMSCalculator:
    """Procedure Evolution Maturity Score (PEMS) calculator"""

    def __init__(self, convergence_threshold: float = 0.01):
        self.convergence_threshold: float = convergence_threshold  # epsilon
        self.history: List[float] = []  # PEMS history records

    def compute(
        self,
        success_rate: float,
        num_proc_nodes: int,
        skill_token_length: int,
        current_embedding: np.ndarray,
        previous_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the PEMS score for a single skill.

        Args:
            success_rate: eta^(k) - average success rate of source episodes
            num_proc_nodes: |V_proc| - total number of procedural skill nodes
            skill_token_length: l^(k) - token length of the skill text
            current_embedding: embedding of the current skill version
            previous_embedding: embedding of the previous skill version (None on first call)

        Returns:
            PEMS score
        """
        # Avoid division by zero: ensure at least 1 procedural node
        num_proc_nodes = max(num_proc_nodes, 1)
        # Ensure token length is at least 1 to prevent log(0)
        skill_token_length = max(skill_token_length, 1)

        # Compute delta: amount of embedding change
        if previous_embedding is not None:
            delta = self.compute_delta(current_embedding, previous_embedding)
        else:
            # No previous version on first call, delta=0, so (1 - 0) = 1
            delta = 0.0

        # PEMS = eta / (|V_proc| * ln(l)) * (1 - delta)
        log_length = math.log(skill_token_length)  # natural logarithm
        denominator = num_proc_nodes * log_length
        if denominator == 0:
            pems = 0.0
        else:
            pems = success_rate / denominator * (1.0 - delta)

        self.history.append(pems)
        return pems

    def compute_delta(
        self, current_embedding: np.ndarray, previous_embedding: np.ndarray
    ) -> float:
        """Compute the difference between two embeddings (1 - cosine_similarity)."""
        # Normalize
        norm_curr = np.linalg.norm(current_embedding)
        norm_prev = np.linalg.norm(previous_embedding)

        if norm_curr == 0 or norm_prev == 0:
            # Cannot compute cosine for a zero vector; return maximum difference
            return 1.0

        cosine_sim = float(
            np.dot(current_embedding, previous_embedding) / (norm_curr * norm_prev)
        )
        # Clamp to [-1, 1] range (floating-point error protection)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return 1.0 - cosine_sim

    def has_converged(self, current_pems: float) -> bool:
        """Check whether PEMS has converged: delta_PEMS < epsilon.

        Requires at least 2 history records to determine convergence.
        """
        if len(self.history) < 2:
            return False
        previous_pems = self.history[-2]
        delta_pems = abs(current_pems - previous_pems)
        return delta_pems < self.convergence_threshold

    def reset(self) -> None:
        """Reset the history records."""
        self.history = []
