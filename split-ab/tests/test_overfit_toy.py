"""
Test 1: Overfitting on Toy Cluster (tests/test_overfit_toy.py).

Validates that the GNN + MADDPG learning core can overfit and solve the 2-node
no-stress mock cluster within a small number of episodes, reliably choosing
the node with vastly more free capacity (Node 1).
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


def test_overfit_toy_environment():
    """Verify that agent learns to pick the superior node on the 2-node toy cluster."""
    # Fix seed for test determinism
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")
    env = MockClusterEnv(tiny=True, max_steps_per_episode=20, seed=42)

    # Fast convergence hyperparameters for toy cluster
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
        use_gnn=True,
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
    noise = GaussianNoise(sigma_start=0.25, sigma_end=0.02, decay_steps=400)

    episodes = 35
    warmup_steps = 40
    batch_size = 32
    global_step = 0
    episode_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            global_step += 1
            if global_step < warmup_steps:
                actions = np.random.uniform(0.0, 1.0, size=(2, 3)).astype(np.float32)
            else:
                act_noise = noise.sample((2, 3))
                actions = trainer.select_actions(obs, noise=act_noise, eval_mode=False)

            next_obs, reward, done, info = env.step(actions)
            buffer.push(obs, actions, reward, next_obs, done, selected_node=info["selected_node"])
            obs = next_obs
            ep_reward += reward

            if global_step >= warmup_steps and len(buffer) >= batch_size:
                noise.step()
                trainer.update(buffer, batch_size=batch_size)

        avg_step_reward = ep_reward / env.max_steps_per_episode
        episode_rewards.append(avg_step_reward)

    # Evaluate deterministically on 10 test episodes
    eval_rewards = []
    selected_nodes = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        ep_eval_rew = 0.0
        while not done:
            actions = trainer.select_actions(obs, eval_mode=True)
            next_obs, reward, done, info = env.step(actions)
            ep_eval_rew += reward
            selected_nodes.append(info["selected_node"])
            obs = next_obs
        eval_rewards.append(ep_eval_rew / env.max_steps_per_episode)

    mean_eval_reward = float(np.mean(eval_rewards))
    pct_node_1 = sum(1 for n in selected_nodes if n == 1) / len(selected_nodes)

    print(f"\n[Test Toy Overfit] Evaluation Mean Reward: {mean_eval_reward:.3f}")
    print(f"[Test Toy Overfit] Optimal Node (Node 1) Selection Rate: {pct_node_1 * 100:.1f}%")

    # Assertions
    # 1. Optimal reward is 1.0; assert eval reward is close to optimal (> 0.85)
    assert mean_eval_reward >= 0.85, (
        f"Agent failed to reach threshold near optimal! Mean eval reward: {mean_eval_reward:.3f} < 0.85"
    )
    # 2. Assert agent reliably selects Node 1 (more than 90% of decisions)
    assert pct_node_1 >= 0.90, (
        f"Agent did not reliably select the node with more capacity! Selected Node 1: {pct_node_1 * 100:.1f}%"
    )
