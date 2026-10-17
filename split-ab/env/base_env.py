"""
Base Environment Interface for Cluster Scheduling.

This defines the abstract interface that any cluster simulator (including
Split A's future production simulator and Split B's mock simulator) must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np


class BaseClusterEnv(ABC):
    """Abstract cluster environment interface.

    Any cluster simulator must adhere to this Gym-style interface so that
    Split B's GNN + MADDPG learning core can interface with it seamlessly.
    """

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the cluster environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            obs: Observation dict containing:
                - 'x': np.ndarray of shape (N, 12), float32 node features
                - 'edge_index': np.ndarray of shape (2, E), int64 graph connectivity
                - 'edge_attr': np.ndarray of shape (E, 1), float32 edge attributes (e.g., same_zone)
                - 'pod_features': np.ndarray of shape (6,), float32 incoming pod request features
                - 'num_nodes': int, number of nodes N in the cluster
            info: Metadata dictionary with auxiliary state information.
        """
        pass

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one scheduling step in the cluster.

        Args:
            actions: np.ndarray of shape (N, 3), values in [0, 1].
                For each node i: [ft_suitability, util_suitability, cost_suitability].

        Returns:
            next_obs: Observation dict for the next step, matching reset() structure.
            reward: Single scalar float, weighted combination of FT, Util, and Cost signals.
            done: Boolean flag indicating if the episode has terminated.
            info: Dictionary containing unweighted raw reward signals and scheduling details:
                - 'fault_tolerance_signal': float
                - 'utilization_signal': float
                - 'cost_signal': float
                - 'selected_node': int or None
                - 'feasible': bool
        """
        pass

    @abstractmethod
    def observation_spec(self) -> Dict[str, Any]:
        """Return the observation specification (shapes and data types)."""
        pass

    @abstractmethod
    def action_spec(self) -> Dict[str, Any]:
        """Return the action specification (shapes and bounds)."""
        pass
