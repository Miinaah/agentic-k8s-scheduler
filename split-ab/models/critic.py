"""
Centralized Permutation-Invariant Critic for MADDPG.

Shared centralized critic used exclusively during training.
Computes Q-values based on per-node embedding & action, mean-pooled cluster
context across all nodes, and incoming pod request features.
Permutation-invariant across cluster size variations.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralizedCritic(nn.Module):
    """Centralized Critic evaluating scheduling decisions across cluster nodes.

    Architecture:
    Linear(203 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 1)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        action_dim: int = 3,
        pod_dim: int = 6,
        input_dim: int = 203,
    ):
        """Initialize CentralizedCritic.

        Args:
            embed_dim: Node embedding dimension (default 64).
            action_dim: Action score dimension per node (default 3).
            pod_dim: Incoming pod feature dimension (default 6).
            input_dim: Specified input dimension to the first linear layer (default 203).

        Design Note on input_dim=203:
            The raw feature components are:
            own_embedding (64) + own_action (3) + pooled_pairs (67) + pod_features (6) = 140 dims.
            To honor the exact prompt specification of Linear(203 -> 256), vectors are
            zero-padded from 140 to 203 dims when input_dim=203.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.pod_dim = pod_dim
        self.pair_dim = embed_dim + action_dim  # 64 + 3 = 67
        self.raw_feature_dim = self.pair_dim + self.pair_dim + pod_dim  # 67 + 67 + 6 = 140
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Linear output for Q-value
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        actions: torch.Tensor,
        pod_features: torch.Tensor,
        node_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Q-value(s).

        Args:
            node_embeddings: Tensor of shape (B, N, 64) or (N, 64).
            actions: Tensor of shape (B, N, 3) or (N, 3).
            pod_features: Tensor of shape (B, 6) or (6,).
            node_idx: Optional integer node index to evaluate. If None, returns Q for all nodes.

        Returns:
            Tensor of shape (B, N, 1) or (B, 1) containing scalar Q-values.
        """
        # Ensure 3D batch shape: (B, N, D)
        single_env = False
        if node_embeddings.dim() == 2:
            single_env = True
            node_embeddings = node_embeddings.unsqueeze(0)  # (1, N, 64)
            actions = actions.unsqueeze(0)                  # (1, N, 3)
            pod_features = pod_features.unsqueeze(0)        # (1, 6)

        B, N, _ = node_embeddings.shape

        # 1. per node j: pair_j = embedding_j (64) concat action_j (3) -> 67 dims
        pairs = torch.cat([node_embeddings, actions], dim=-1)  # (B, N, 67)

        # 2. pooled = mean over j of pair_j -> 67 dims
        pooled = pairs.mean(dim=1, keepdim=True)  # (B, 1, 67)
        pooled_expanded = pooled.expand(-1, N, -1)  # (B, N, 67)

        # Expand pod features: (B, 6) -> (B, N, 6)
        pod_expanded = pod_features.unsqueeze(1).expand(-1, N, -1)

        # 3. critic_input = own_embedding(64) concat own_action(3) concat pooled(67) concat pod_features(6) = 140 dims
        critic_input = torch.cat(
            [node_embeddings, actions, pooled_expanded, pod_expanded], dim=-1
        )  # (B, N, 140)

        # 4. Zero-pad to specified input_dim (203) if configured
        if critic_input.size(-1) < self.input_dim:
            pad_amount = self.input_dim - critic_input.size(-1)
            critic_input = F.pad(critic_input, (0, pad_amount))

        # 5. Pass through MLP: (B, N, input_dim) -> (B, N, 1)
        q_vals = self.net(critic_input)

        if node_idx is not None:
            q_vals = q_vals[:, node_idx : node_idx + 1, :]  # (B, 1, 1)
            if single_env:
                return q_vals.squeeze(0).squeeze(-1)  # (1,)
            return q_vals.squeeze(1)  # (B, 1)

        if single_env:
            return q_vals.squeeze(0)  # (N, 1)
        return q_vals
