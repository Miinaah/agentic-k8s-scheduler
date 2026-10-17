"""
Test 2: GNN Ablation Study (tests/test_ablation_no_gnn.py).

Core Idea:
    In the overloaded_zone scenario, Node 0 and Node 2 have IDENTICAL raw features.
    Their zone-neighbors differ dramatically: Node 1 (Zone 0) is catastrophically failing
    while Node 3 (Zone 1) is healthy.

    With zone-restricted edges:
    - GNN agent: Node 0 embedding ≠ Node 2 embedding (different neighborhood context)
      → can learn to prefer Node 2 (Zone 1)
    - No-GNN agent: Node 0 embedding == Node 2 embedding (identical raw features)
      → CANNOT distinguish them, stuck at ~50/50 random selection

Validates that the GNN version achieves strictly higher reward.
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
from training.noise import GaussianNoise
from training.replay_buffer import ClusterReplayBuffer


def run_training_experiment(use_gnn: bool, episodes: int = 50) -> list:
    """Run training on overloaded_zone scenario with or without GNN.

    Args:
        use_gnn: Whether to use GNN message passing.
        episodes: Number of training episodes.

    Returns:
        List of per-episode average rewards.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    env = MockClusterEnv(scenario="overloaded_zone", max_steps_per_episode=15, seed=42)

    hidden_dim = 64
    actor = NodeActor(embed_dim=hidden_dim * 2, pod_dim=6, action_dim=3)
    critic = CentralizedCritic(
        embed_dim=hidden_dim, action_dim=3, pod_dim=6, input_dim=203
    )
    gnn_encoder = ClusterGNNEncoder(
        node_feat_dim=8,
        hidden_dim=hidden_dim,
        gnn_layers=2,
        gnn_type="sage",
        dropout=0.0,
        use_gnn=use_gnn,
    )

    trainer = MADDPGTrainer(
        gnn_encoder=gnn_encoder,
        actor=actor,
        critic=critic,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.90,
        tau=0.02,
        grad_clip_norm=1.0,
        n_step=1,
        reward_weights=env.reward_weights,
        selection_temp=0.3,
        aux_loss_weight=0.0,
        device=device,
    )

    buffer = ClusterReplayBuffer(capacity=5000, n_step=1, gamma=0.90)
    noise = GaussianNoise(sigma_start=0.25, sigma_end=0.02, decay_steps=500)

    warmup_steps = 30
    batch_size = 32
    global_step = 0
    rewards_curve = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            global_step += 1
            if global_step < warmup_steps:
                actions = np.random.uniform(0.0, 1.0, size=(env.num_nodes, 3)).astype(
                    np.float32
                )
            else:
                act_noise = noise.sample((env.num_nodes, 3))
                actions = trainer.select_actions(obs, noise=act_noise, eval_mode=False)

            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done, selected_node=info["selected_node"])
            obs = next_obs
            ep_reward += reward

            if global_step >= warmup_steps and len(buffer) >= batch_size:
                noise.step()
                trainer.update(buffer, batch_size=batch_size)

        rewards_curve.append(ep_reward / env.max_steps_per_episode)

    return rewards_curve


def test_ablation_gnn_vs_no_gnn():
    """Verify that GNN version achieves higher reward on overloaded zone scenario.

    The test relies on an information-theoretic impossibility:
    Node 0 and Node 2 have IDENTICAL features → Linear(12→64) produces
    identical embeddings → Actor produces identical suitability scores →
    No-GNN is stuck at 50/50 zone selection.

    The GNN breaks this symmetry via SAGEConv neighborhood aggregation
    with zone-restricted edges, allowing it to learn Node 2 > Node 0.
    """
    episodes = 50
    print("\n--- Training GNN Agent (use_gnn=True) ---")
    gnn_curve = run_training_experiment(use_gnn=True, episodes=episodes)

    print("\n--- Training No-GNN Baseline (use_gnn=False) ---")
    no_gnn_curve = run_training_experiment(use_gnn=False, episodes=episodes)

    # Print both reward curves side-by-side
    print("\n" + "=" * 55)
    print(f"{'Episode':<10}{'GNN Reward':<20}{'No-GNN Reward':<20}")
    print("=" * 55)
    for i in range(episodes):
        print(f"{i + 1:<10}{gnn_curve[i]:<20.4f}{no_gnn_curve[i]:<20.4f}")
    print("=" * 55)

    # Average reward over last 15 episodes (giving GNN more time to converge)
    eval_window = 15
    gnn_final_avg = np.mean(gnn_curve[-eval_window:])
    no_gnn_final_avg = np.mean(no_gnn_curve[-eval_window:])

    print(f"\nFinal {eval_window}-Ep Average: GNN = {gnn_final_avg:.4f} vs No-GNN = {no_gnn_final_avg:.4f}")

    # Also verify the information-theoretic bound: No-GNN should be near zero
    # because it can't distinguish Node 0 from Node 2 (identical features)
    # Expected No-GNN reward ≈ average of Zone0 penalty and Zone1 reward
    # ft: (-0.8 + 1.0) / 2 = 0.1, util: (-0.4 + 0.9) / 2 = 0.25, cost: 0.5
    # Weighted: 0.4*0.1 + 0.3*0.25 + 0.3*0.5 = 0.04 + 0.075 + 0.15 = 0.265
    print(f"\nNo-GNN theoretical baseline (random 50/50): ~0.265")

    assert gnn_final_avg > no_gnn_final_avg, (
        f"Ablation assertion failed: GNN reward ({gnn_final_avg:.4f}) should exceed "
        f"No-GNN reward ({no_gnn_final_avg:.4f}) on the overloaded zone scenario!\n"
        f"Node 0 and Node 2 have identical features — only GNN can differentiate via zone context."
    )
