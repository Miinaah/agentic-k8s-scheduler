"""
MADDPG Trainer for Cluster Scheduling.

Implements centralized training with decentralized execution (CTDE):
- GNN encoder embeds cluster state
- Shared actor outputs suitability actions per node
- Centralized critic evaluates global cluster actions
- Polyak soft target network updates
- Gradient clipping and n-step discounted Bellman targets
"""

import copy
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor import NodeActor
from models.critic import CentralizedCritic
from models.gnn_encoder import ClusterGNNEncoder
from training.replay_buffer import ClusterReplayBuffer


class MADDPGTrainer:
    """Trainer orchestrating GNN, Actor, and Critic training using MADDPG."""

    def __init__(
        self,
        gnn_encoder: ClusterGNNEncoder,
        actor: NodeActor,
        critic: CentralizedCritic,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        grad_clip_norm: float = 1.0,
        weight_decay: float = 0.0,
        n_step: int = 3,
        reward_weights: Optional[Dict[str, float]] = None,
        selection_temp: float = 0.1,
        aux_loss_weight: float = 0.5,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    ):
        """Initialize MADDPG trainer."""
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        self.n_step = n_step
        self.selection_temp = selection_temp
        self.aux_loss_weight = aux_loss_weight

        if reward_weights is None:
            self.reward_weights = {"fault_tolerance": 0.4, "utilization": 0.3, "cost": 0.3}
        else:
            self.reward_weights = reward_weights

        self.w_ft = float(self.reward_weights.get("fault_tolerance", 0.4))
        self.w_util = float(self.reward_weights.get("utilization", 0.3))
        self.w_cost = float(self.reward_weights.get("cost", 0.3))

        # Core networks
        self.gnn_encoder = gnn_encoder.to(device)
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        # Target networks (deep copies)
        self.gnn_encoder_target = copy.deepcopy(self.gnn_encoder).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Freeze target params from direct optimization
        for p in self.gnn_encoder_target.parameters():
            p.requires_grad = False
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        # Actor optimizer updates both actor and GNN encoder representations
        self.actor_params = list(self.actor.parameters()) + list(self.gnn_encoder.parameters())
        self.actor_optimizer = torch.optim.Adam(
            self.actor_params, lr=actor_lr, weight_decay=weight_decay
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=weight_decay
        )

    def select_actions(
        self,
        obs: Dict[str, Any],
        noise: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select actions for all cluster nodes given current observation.

        Args:
            obs: Observation dict from environment.
            noise: Optional exploration noise array of shape (N, 3).
            eval_mode: If True, executes deterministically without gradients.

        Returns:
            np.ndarray of shape (N, 3) in range [0, 1].
        """
        self.gnn_encoder.eval()
        self.actor.eval()

        with torch.no_grad():
            x = torch.as_tensor(obs["x"], dtype=torch.float32, device=self.device)
            edge_index = torch.as_tensor(
                obs["edge_index"], dtype=torch.int64, device=self.device
            )
            pod = torch.as_tensor(
                obs["pod_features"], dtype=torch.float32, device=self.device
            )

            # Node embeddings
            embeddings = self.gnn_encoder(x, edge_index=edge_index)  # (N, 64)

            # Actor action scores
            actions = self.actor(torch.cat([embeddings, self.gnn_encoder.input_proj(x)], dim=-1), pod).cpu().numpy()  # (N, 3)

        if not eval_mode and noise is not None:
            actions = actions + noise
            actions = np.clip(actions, 0.0, 1.0)

        return actions.astype(np.float32)

    def update(
        self, replay_buffer: ClusterReplayBuffer, batch_size: int = 128
    ) -> Dict[str, float]:
        """Perform one training update step using transitions sampled from replay buffer."""
        if len(replay_buffer) < batch_size:
            return {}

        self.gnn_encoder.train()
        self.actor.train()
        self.critic.train()

        (
            obs_batch,
            actions,
            rewards,
            next_obs_batch,
            dones,
            pod_features,
            next_pod_features,
            selected_nodes,
            node_signals,
        ) = replay_buffer.sample(batch_size, device=self.device)
        replay_buffer_sample = [node_signals]

        B = obs_batch.num_graphs
        N = obs_batch.num_nodes // B
        selected_nodes = torch.clamp(selected_nodes, 0, N - 1)

        # -------------------------------------------------------------
        # 1. Compute Centralized Target Q
        # -------------------------------------------------------------
        with torch.no_grad():
            self.gnn_encoder_target.eval(); self.actor_target.eval(); self.critic_target.eval()
            # Target GNN embeddings
            next_embeddings = self.gnn_encoder_target(next_obs_batch).view(B, N, -1)
            next_h0 = self.gnn_encoder_target.input_proj(next_obs_batch.x).view(B, N, -1)

            # Target decentralized actor actions
            next_actions = self.actor_target(torch.cat([next_embeddings, next_h0], dim=-1), next_pod_features)

            # Target centralized critic evaluation
            target_q_all = self.critic_target(
                next_embeddings, next_actions, next_pod_features
            ).squeeze(-1)  # (B, N)

            # Next selection preference
            next_suitability = (
                self.w_ft * next_actions[..., 0]
                + self.w_util * next_actions[..., 1]
                + self.w_cost * next_actions[..., 2]
            )
            next_selected = next_suitability.argmax(dim=1)
            target_q_selected = target_q_all[torch.arange(B), next_selected].unsqueeze(1)  # (B, 1)

            # N-step Bellman target
            discount = self.gamma ** self.n_step
            target_value = rewards + discount * (1.0 - dones) * target_q_selected

        # -------------------------------------------------------------
        # 2. Update Centralized Critic
        # -------------------------------------------------------------
        current_embeddings_detached = self.gnn_encoder(obs_batch).detach().view(B, N, -1)
        current_q_all = self.critic(
            current_embeddings_detached, actions, pod_features
        ).squeeze(-1)  # (B, N)

        # Evaluate the chosen node's predicted value against the observed return
        current_q_selected = current_q_all[torch.arange(B), selected_nodes].unsqueeze(1)  # (B, 1)

        critic_loss = F.mse_loss(current_q_selected, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # -------------------------------------------------------------
        # 3. Update Actor (and GNN Encoder via actor gradient)
        # -------------------------------------------------------------
        current_embeddings = self.gnn_encoder(obs_batch).view(B, N, -1)
        current_h0 = self.gnn_encoder.input_proj(obs_batch.x).view(B, N, -1)
        cat_embeddings = torch.cat([current_embeddings, current_h0], dim=-1)
        pred_actions = self.actor(cat_embeddings, pod_features)  # (B, N, 3)

        # Q-values under predicted actions (detach pred_actions so actor updates focus on node selection softmax)
        actor_q_all = self.critic(current_embeddings.detach(), pred_actions.detach(), pod_features).squeeze(-1)  # (B, N)

        # Softmax suitability weighting approximates cluster scheduling resolution
        suitability = (
            self.w_ft * pred_actions[..., 0]
            + self.w_util * pred_actions[..., 1]
            + self.w_cost * pred_actions[..., 2]
        )  # (B, N)
        selection_weights = F.softmax(suitability / self.selection_temp, dim=1)  # (B, N)

        # Expected Q-value of the cluster scheduling decision
        expected_cluster_q = (selection_weights * actor_q_all).sum(dim=1)  # (B,)
        actor_loss = -expected_cluster_q.mean()
        
        # Auxiliary loss for meaningful heads
        if self.aux_loss_weight > 0:
            node_signals = replay_buffer_sample[-1]
            aux = F.mse_loss(pred_actions, (node_signals + 1) / 2)
            actor_loss = actor_loss + self.aux_loss_weight * aux

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip_norm)
        self.actor_optimizer.step()

        # -------------------------------------------------------------
        # 4. Soft Target Updates (Polyak Averaging)
        # -------------------------------------------------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.gnn_encoder, self.gnn_encoder_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_value": float(current_q_selected.mean().item()),
        }


    def _soft_update(self, source_net: nn.Module, target_net: nn.Module):
        """Soft update target model parameters: target = tau * source + (1 - tau) * target."""
        for param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str):
        """Save model checkpoints to file."""
        state = {
            "gnn_encoder": self.gnn_encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": {
                "node_feat_dim": getattr(self.gnn_encoder, "node_feat_dim", 8),
                "hidden_dim": getattr(self.gnn_encoder, "hidden_dim", 64),
                "gnn_layers": getattr(self.gnn_encoder, "gnn_layers", 2),
                "gnn_type": getattr(self.gnn_encoder, "gnn_type", "sage"),
            }
        }
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoints from file."""
        state = torch.load(filepath, map_location=self.device)
        self.gnn_encoder.load_state_dict(state["gnn_encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.gnn_encoder_target.load_state_dict(state["gnn_encoder"])
        self.actor_target.load_state_dict(state["actor"])
        self.critic_target.load_state_dict(state["critic"])
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
