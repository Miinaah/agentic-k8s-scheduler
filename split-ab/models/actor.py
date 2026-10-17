"""
Actor Network for Node Scheduling Suitability.

Shared homogeneous actor executed in a decentralized fashion on each cluster node.
Takes per-node contextual embedding (64) concatenated with incoming pod request (6)
and outputs suitability scores for Fault Tolerance, Utilization, and Cost in [0, 1].
"""

from typing import Union
import torch
import torch.nn as nn


class NodeActor(nn.Module):
    """Homogeneous actor shared across all cluster nodes.

    Architecture:
    Linear(70 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 3) -> Sigmoid
    """

    def __init__(
        self,
        embed_dim: int = 64,
        pod_dim: int = 6,
        action_dim: int = 3,
    ):
        """Initialize NodeActor.

        Args:
            embed_dim: Dimension of node embedding from GNN (default 64).
            pod_dim: Dimension of incoming pod features (default 6).
            action_dim: Dimension of action space (default 3: FT, Util, Cost).
        """
        super().__init__()
        self.input_dim = embed_dim + pod_dim  # 64 + 6 = 70
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Sigmoid(),  # Bound output scores in [0, 1]
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        pod_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass to compute action suitability scores.

        Args:
            node_embeddings: Shape (N, 64) or (B, N, 64) or (B * N, 64).
            pod_features: Shape (6,) or (B, 6) or (B * N, 6).

        Returns:
            Actions tensor of shape (N, 3) or (B, N, 3) or (B * N, 3) in range [0, 1].
        """
        # Case 1: node_embeddings is (N, 64) and pod_features is (6,)
        if node_embeddings.dim() == 2 and pod_features.dim() == 1:
            N = node_embeddings.size(0)
            pod_expanded = pod_features.unsqueeze(0).expand(N, -1)
            actor_input = torch.cat([node_embeddings, pod_expanded], dim=-1)
            return self.net(actor_input)

        # Case 2: node_embeddings is (B, N, 64) and pod_features is (B, 6)
        elif node_embeddings.dim() == 3 and pod_features.dim() == 2:
            B, N, _ = node_embeddings.shape
            pod_expanded = pod_features.unsqueeze(1).expand(-1, N, -1)
            actor_input = torch.cat([node_embeddings, pod_expanded], dim=-1)
            return self.net(actor_input)

        # Case 3: already broadcasted or flattened (M, 64) and (M, 6)
        elif node_embeddings.dim() == 2 and pod_features.dim() == 2:
            if node_embeddings.size(0) != pod_features.size(0):
                # Attempt to expand pod across nodes if B matches
                raise ValueError(
                    f"Mismatched batch sizes: {node_embeddings.shape} vs {pod_features.shape}"
                )
            actor_input = torch.cat([node_embeddings, pod_features], dim=-1)
            return self.net(actor_input)

        else:
            raise ValueError(
                f"Unsupported input dimensions: embeddings {node_embeddings.shape}, pod {pod_features.shape}"
            )
