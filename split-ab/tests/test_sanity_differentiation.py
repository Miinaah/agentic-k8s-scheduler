"""
Test 3: Sanity Differentiation Check (tests/test_sanity_differentiation.py).

Validates that after a short training run, when fed a cluster state containing
heterogeneous nodes (e.g. idle vs overloaded vs degraded), the GNN encoder
and Actor produce distinctly differentiated suitability scores rather than
collapsing to identical or uniform outputs across nodes.
"""

import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.mock_cluster_env import MockClusterEnv
from models.actor import NodeActor
from models.critic import CentralizedCritic
from models.gnn_encoder import ClusterGNNEncoder
from training.maddpg_trainer import MADDPGTrainer
from training.replay_buffer import ClusterReplayBuffer


def test_actor_node_differentiation():
    """Verify that actor outputs distinct scores for distinctly different node states."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    # 1. Run a brief 10-episode training run to populate weights with gradient updates
    env = MockClusterEnv(num_nodes=5, num_zones=2, max_steps_per_episode=10, seed=42)
    hidden_dim = 64
    actor = NodeActor(embed_dim=hidden_dim * 2, pod_dim=6, action_dim=3)
    critic = CentralizedCritic(embed_dim=hidden_dim, action_dim=3, pod_dim=6, input_dim=203)
    gnn_encoder = ClusterGNNEncoder(
        node_feat_dim=8,
        hidden_dim=hidden_dim,
        gnn_layers=2,
        gnn_type="sage",
        dropout=0.0,
        use_gnn=True,
    )

    trainer = MADDPGTrainer(
        gnn_encoder=gnn_encoder,
        actor=actor,
        critic=critic,
        actor_lr=1e-3,
        critic_lr=1e-3,
        reward_weights=env.reward_weights,
        selection_temp=0.1,
        device=device,
    )

    buffer = ClusterReplayBuffer(capacity=1000, n_step=1)

    for ep in range(12):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = trainer.select_actions(obs, noise=np.random.normal(0, 0.1, size=(5, 3)))
            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done, selected_node=info["selected_node"])
            obs = next_obs
            if len(buffer) >= 32:
                trainer.update(buffer, batch_size=32)

    # 2. Construct a synthetic cluster batch with 4 clearly disparate node states:
    # Node 0: Ideal node (low util 0.10, perfect health 0.99, low cost 0.10)
    # Node 1: Saturated/failing node (high util 0.98, near-dead health 0.10, high cost 0.90)
    # Node 2: Moderate node (util 0.50, health 0.80, cost 0.50)
    # Node 3: Idle but unproven/failing node (util 0.05, health 0.15, cost 0.80)
    distinct_nodes = np.array(
        [
            [0.10, 0.10, 0.10, 0.10, 0.99, 0.10, 0.0, 0.0],
            [0.98, 0.98, 0.95, 0.90, 0.10, 0.90, 0.0, 0.8],
            [0.50, 0.50, 0.50, 0.40, 0.80, 0.50, 0.5, 0.3],
            [0.05, 0.05, 0.05, 0.05, 0.15, 0.80, 0.5, 0.0],
        ],
        dtype=np.float32,
    )

    # Fully connected edges between 4 nodes
    src, dst = [], []
    for i in range(4):
        for j in range(4):
            if i != j:
                src.append(i)
                dst.append(j)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = np.zeros((len(src), 1), dtype=np.float32)
    pod_features = np.array([0.15, 0.15, 0.10, 1.0, 1.0, 0.5], dtype=np.float32)

    eval_obs = {
        "x": distinct_nodes,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pod_features": pod_features,
        "num_nodes": 4,
    }

    # 3. Generate actions without noise (deterministic evaluation)
    actions = trainer.select_actions(eval_obs, eval_mode=True)

    print("\n[Sanity Differentiation Check] Evaluated Action Scores:")
    for i, name in enumerate(["Node 0 (Optimal)", "Node 1 (Saturated)", "Node 2 (Moderate)", "Node 3 (Degraded)"]):
        print(f"  {name}: FT={actions[i, 0]:.4f}, Util={actions[i, 1]:.4f}, Cost={actions[i, 2]:.4f}")

    # 4. Measure dispersion across nodes
    # Score standard deviation across the 4 nodes for each action dimension
    stds = np.std(actions, axis=0)
    print(f"[Sanity Differentiation Check] Action Stds across nodes: FT={stds[0]:.4f}, Util={stds[1]:.4f}, Cost={stds[2]:.4f}")

    # Maximum pairwise difference between any two nodes
    max_diff_ft = np.max(actions[:, 0]) - np.min(actions[:, 0])
    max_diff_util = np.max(actions[:, 1]) - np.min(actions[:, 1])
    max_diff_cost = np.max(actions[:, 2]) - np.min(actions[:, 2])

    print(f"[Sanity Differentiation Check] Max differences: FT={max_diff_ft:.4f}, Util={max_diff_util:.4f}, Cost={max_diff_cost:.4f}")

    # Assert that the network has not collapsed to uniform scores
    assert np.mean(stds) > 0.01, (
        f"GNN/Actor output collapsed! Mean standard deviation across nodes is too low: {np.mean(stds):.5f} <= 0.01"
    )
    assert max(max_diff_ft, max_diff_util, max_diff_cost) > 0.03, (
        "Max difference across heterogeneous nodes is too small; actor is failing to differentiate cluster states!"
    )
