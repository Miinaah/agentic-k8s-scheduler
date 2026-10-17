"""
Training Entry Point for GNN + MADDPG Cluster Scheduler.

Usage:
    python training/train.py --config configs/default.yaml
    python training/train.py --config configs/default.yaml --tiny
    python training/train.py --config configs/default.yaml --episodes 100 --use_gnn
"""

import argparse
import os
import sys
from typing import Any, Dict
import numpy as np
import torch
import yaml
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.wrapped_cluster_env import WrappedClusterEnv
from models.actor import NodeActor
from models.critic import CentralizedCritic
from models.gnn_encoder import ClusterGNNEncoder
from training.maddpg_trainer import MADDPGTrainer
from training.noise import create_noise
from training.replay_buffer import ClusterReplayBuffer
from utils.logger import MetricLogger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with overrides for config parameters."""
    parser = argparse.ArgumentParser(
        description="Train GNN + MADDPG Cluster Scheduler (Split B)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Run in 2-node no-stress toy cluster mode for quick overfit testing",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override total number of training episodes",
    )
    parser.add_argument(
        "--use_gnn",
        dest="use_gnn",
        action="store_true",
        default=None,
        help="Enable GNN message passing",
    )
    parser.add_argument(
        "--no_gnn",
        dest="use_gnn",
        action="store_false",
        help="Disable GNN message passing (ablation study)",
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        default=None,
        choices=["sage", "gat"],
        help="GNN layer type ('sage' or 'gat')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on (auto detects CUDA -> MPS -> CPU)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="default",
        choices=["default", "tiny", "overloaded_zone"],
        help="Cluster scenario to simulate",
    )
    return parser.parse_args()


def train(config: Dict[str, Any], seed: int = 42):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    """Main training routine."""
    # Device setup: Auto detects CUDA -> Apple Silicon MPS -> CPU
    device_str = config.get("device", "auto")
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"[*] Training on device: {device}")

    # Initialize environment
    scenario = "tiny" if config.get("tiny", False) else config.get("scenario", "default")
    env = WrappedClusterEnv(
        seed=seed,
        num_nodes=config.get("num_nodes", 10),
        num_zones=config.get("num_zones", 3),
        reward_weights=config.get("reward_weights", None),
        max_steps_per_episode=config.get("max_steps_per_episode", 50),
        tiny=config.get("tiny", False),
        scenario=scenario,
    )
    print(
        f"[*] Initialized MockClusterEnv (Scenario: {scenario}, Nodes: {env.num_nodes}, Zones: {env.num_zones})"
    )

    # Initialize GNN Encoder
    gnn_encoder = ClusterGNNEncoder(
        node_feat_dim=config.get("node_feat_dim", 12),
        hidden_dim=config.get("hidden_dim", 64),
        gnn_layers=config.get("gnn_layers", 2),
        gnn_type=config.get("gnn_type", "sage"),
        gat_heads=config.get("gat_heads", 4),
        dropout=config.get("dropout", 0.1),
        residual=config.get("residual", False),
        use_gnn=config.get("use_gnn", True),
    )

    # Initialize Actor
    actor = NodeActor(
        embed_dim=config.get("hidden_dim", 64) * 2,
        pod_dim=config.get("pod_feat_dim", 6),
        action_dim=config.get("action_dim", 3),
    )

    # Initialize Centralized Critic
    critic = CentralizedCritic(
        embed_dim=config.get("hidden_dim", 64),
        action_dim=config.get("action_dim", 3),
        pod_dim=config.get("pod_feat_dim", 6),
        input_dim=config.get("critic_input_dim", 203),
    )

    # Exploration Noise
    noise = create_noise(
        noise_type=config.get("noise_type", "gaussian"),
        sigma_start=config.get("noise_sigma_start", 0.3),
        sigma_end=config.get("noise_sigma_end", 0.05),
        decay_steps=config.get("noise_decay_steps", 50000),
        ou_theta=config.get("ou_theta", 0.15),
        ou_sigma=config.get("ou_sigma", 0.2),
    )

    # Replay Buffer
    replay_buffer = ClusterReplayBuffer(
        capacity=config.get("buffer_size", 100000),
        n_step=config.get("n_step_return", 3),
        gamma=config.get("gamma", 0.95),
    )

    # MADDPG Trainer
    trainer = MADDPGTrainer(
        gnn_encoder=gnn_encoder,
        actor=actor,
        critic=critic,
        actor_lr=float(config.get("actor_lr", 1e-4)),
        critic_lr=float(config.get("critic_lr", 1e-3)),
        gamma=float(config.get("gamma", 0.95)),
        tau=float(config.get("tau", 0.01)),
        grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        n_step=int(config.get("n_step_return", 3)),
        reward_weights=config.get("reward_weights", None),
        device=device,
        selection_temp=config.get("selection_temp", 0.1),
        aux_loss_weight=float(config.get("aux_loss_weight", 0.5)),
    )

    # Logger
    timestamp = int(time.time())
    run_name = f"{'tiny' if config.get('tiny') else 'cluster'}_{config.get('gnn_type', 'sage')}_{'gnn' if config.get('use_gnn', True) else 'nognn'}_{seed}_{timestamp}"
    logger = MetricLogger(log_dir=config.get("log_dir", "runs"), run_name=run_name)

    total_episodes = config.get("episodes", 200)
    warmup_steps = config.get("warmup_steps", 2000)
    batch_size = config.get("batch_size", 128)
    grad_updates_per_step = config.get("grad_updates_per_step", 1)

    global_step = 0
    print(f"[*] Starting training for {total_episodes} episodes...")

    for ep in range(1, total_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        ft_signals = []
        util_signals = []
        cost_signals = []
        actor_losses = []
        critic_losses = []

        done = False
        while not done:
            global_step += 1

            # Warm-up phase with uniform random exploration
            if global_step < warmup_steps:
                actions = np.random.uniform(0.0, 1.0, size=(env.num_nodes, 3)).astype(
                    np.float32
                )
            else:
                action_noise = noise.sample((env.num_nodes, 3))
                actions = trainer.select_actions(obs, noise=action_noise, eval_mode=False)

            next_obs, reward, done, info = env.step(actions)

            # Store transition in replay buffer
            replay_buffer.push(
                obs,
                actions,
                reward,
                next_obs,
                done,
                selected_node=info.get("selected_node", 0),
                node_signals=info.get("node_signals", np.zeros((env.num_nodes, 3), dtype=np.float32)),
            )
            obs = next_obs

            episode_reward += reward
            ft_signals.append(info["fault_tolerance_signal"])
            util_signals.append(info["utilization_signal"])
            cost_signals.append(info["cost_signal"])

            # Model optimization
            if global_step >= warmup_steps:
                noise.step()
                for _ in range(grad_updates_per_step):
                    update_info = trainer.update(replay_buffer, batch_size=batch_size)
                    if update_info:
                        actor_losses.append(update_info["actor_loss"])
                        critic_losses.append(update_info["critic_loss"])

        mean_ft = float(np.mean(ft_signals)) if ft_signals else 0.0
        mean_util = float(np.mean(util_signals)) if util_signals else 0.0
        mean_cost = float(np.mean(cost_signals)) if cost_signals else 0.0
        mean_actor_loss = float(np.mean(actor_losses)) if actor_losses else 0.0
        mean_critic_loss = float(np.mean(critic_losses)) if critic_losses else 0.0

        logger.log_episode(
            episode=ep,
            reward=episode_reward,
            ft_signal=mean_ft,
            util_signal=mean_util,
            cost_signal=mean_cost,
            actor_loss=mean_actor_loss,
            critic_loss=mean_critic_loss,
            noise_sigma=noise.get_sigma(),
        )

        if ep % 10 == 0 or ep == total_episodes:
            print(
                f"[Ep {ep:03d}/{total_episodes:03d}] "
                f"Reward: {episode_reward:6.2f} | "
                f"FT: {mean_ft:5.2f} | "
                f"Util: {mean_util:5.2f} | "
                f"Cost: {mean_cost:5.2f} | "
                f"CriticLoss: {mean_critic_loss:6.4f} | "
                f"Noise: {noise.get_sigma():.3f}"
            )

    # Save final artifacts
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"[*] Saved final model weights to: {checkpoint_path}")

    # Also save canonical checkpoint for inference / scorer.py
    canonical_name = f"{'tiny' if config.get('tiny') else 'cluster'}_{config.get('gnn_type', 'sage')}_gnn_final.pt"
    canonical_path = os.path.join(checkpoint_dir, canonical_name)
    trainer.save_checkpoint(canonical_path)
    if not config.get("tiny", False):
        default_path = os.path.join(checkpoint_dir, "cluster_sage_gnn_final.pt")
        if canonical_path != default_path:
            trainer.save_checkpoint(default_path)
    print(f"[*] Canonical checkpoint updated at: {canonical_path}")

    plot_path = logger.save_reward_plot()
    print(f"[*] Reward curve plot saved to: {plot_path}")
    logger.close()
    print("[*] Training completed successfully.")


def main():
    args = parse_args()
    config = load_config(args.config)

    # Command line overrides
    if args.tiny:
        config["tiny"] = True
        config["scenario"] = "tiny"
        config["warmup_steps"] = min(config.get("warmup_steps", 2000), 100)
    if args.scenario != "default":
        config["scenario"] = args.scenario
    if args.episodes is not None:
        config["episodes"] = args.episodes
    if args.use_gnn is not None:
        config["use_gnn"] = args.use_gnn
    if args.gnn_type is not None:
        config["gnn_type"] = args.gnn_type
    if args.device != "auto":
        config["device"] = args.device

    train(config)


if __name__ == "__main__":
    main()
