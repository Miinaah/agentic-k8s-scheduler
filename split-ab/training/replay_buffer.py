"""
Replay Buffer with N-Step Returns and PyTorch Geometric Graph Batching.

Supports storing graph structured cluster states, decentralized actions,
and efficient mini-batch sampling collated as PyG Batch objects.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Batch, Data


class ClusterReplayBuffer:
    """Experience replay buffer with N-step return accumulation."""

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 3,
        gamma: float = 0.95,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            n_step: Multi-step return horizon (default 3).
            gamma: Discount factor (default 0.95).
        """
        self.capacity = capacity
        self.n_step = max(1, n_step)
        self.gamma = gamma

        self.buffer: List[Dict[str, Any]] = []
        self.position = 0
        self.n_step_buffer: deque = deque(maxlen=self.n_step)

    def _create_pyg_data(self, obs: Dict[str, Any]) -> Data:
        """Convert environment obs dict to PyG Data object."""
        x = torch.as_tensor(obs["x"], dtype=torch.float32)
        edge_index = torch.as_tensor(obs["edge_index"], dtype=torch.int64)
        edge_attr = torch.as_tensor(obs["edge_attr"], dtype=torch.float32)
        pod = torch.as_tensor(obs["pod_features"], dtype=torch.float32)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pod=pod, num_nodes=x.size(0))

    def push(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
        selected_node: int = 0,
        node_signals: Optional[np.ndarray] = None,
    ):
        """Push a transition to the n-step buffer and commit when ready."""
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "selected_node": selected_node,
            "node_signals": node_signals,
        }
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step:
            self._commit_n_step_transition()

        if done:
            while len(self.n_step_buffer) > 0:
                self._commit_n_step_transition()
            self.n_step_buffer.clear()

    def _commit_n_step_transition(self):
        """Compute discounted n-step return and store in main buffer."""
        # Accumulated discounted reward
        n_reward = 0.0
        n_done = False
        last_next_obs = self.n_step_buffer[-1]["next_obs"]

        for i, trans in enumerate(self.n_step_buffer):
            n_reward += (self.gamma ** i) * trans["reward"]
            if trans["done"]:
                n_done = True
                last_next_obs = trans["next_obs"]
                break

        first_trans = self.n_step_buffer[0]
        stored_item = {
            "obs": first_trans["obs"],
            "action": np.array(first_trans["action"], dtype=np.float32),
            "reward": float(n_reward),
            "next_obs": last_next_obs,
            "done": bool(n_done),
            "selected_node": int(first_trans.get("selected_node") or 0),
            "node_signals": first_trans.get("node_signals", None),
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(stored_item)
        else:
            self.buffer[self.position] = stored_item
        self.position = (self.position + 1) % self.capacity

        if len(self.n_step_buffer) > 0:
            self.n_step_buffer.popleft()

    def sample(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ) -> Tuple[Batch, torch.Tensor, torch.Tensor, Batch, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions.

        Returns:
            obs_batch: PyG Batch of current cluster graphs
            actions: Tensor of shape (B, N, 3)
            rewards: Tensor of shape (B, 1)
            next_obs_batch: PyG Batch of next cluster graphs
            dones: Tensor of shape (B, 1)
            pod_features: Tensor of shape (B, 6)
            next_pod_features: Tensor of shape (B, 6)
            selected_nodes: LongTensor of shape (B,)
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        obs_list = [self._create_pyg_data(s["obs"]) for s in samples]
        next_obs_list = [self._create_pyg_data(s["next_obs"]) for s in samples]

        obs_batch = Batch.from_data_list(obs_list).to(device)
        next_obs_batch = Batch.from_data_list(next_obs_list).to(device)

        actions = torch.tensor(
            np.array([s["action"] for s in samples]), dtype=torch.float32, device=device
        )
        rewards = torch.tensor(
            np.array([s["reward"] for s in samples]), dtype=torch.float32, device=device
        ).unsqueeze(1)
        dones = torch.tensor(
            np.array([s["done"] for s in samples]), dtype=torch.float32, device=device
        ).unsqueeze(1)

        pod_features = torch.tensor(
            np.array([s["obs"]["pod_features"] for s in samples]),
            dtype=torch.float32,
            device=device,
        )
        next_pod_features = torch.tensor(
            np.array([s["next_obs"]["pod_features"] for s in samples]),
            dtype=torch.float32,
            device=device,
        )
        selected_nodes = torch.tensor(
            np.array([s.get("selected_node", 0) for s in samples]),
            dtype=torch.int64,
            device=device,
        )

        node_signals_list = []
        for s in samples:
            sig = s.get("node_signals")
            if sig is None:
                sig = np.zeros((s["action"].shape[0], 3), dtype=np.float32)
            else:
                sig = np.asarray(sig, dtype=np.float32)
            node_signals_list.append(sig)

        node_signals = torch.tensor(
            np.array(node_signals_list, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        return (
            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
            node_signals,
        )

    def __len__(self) -> int:
        return len(self.buffer)
