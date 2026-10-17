"""
Logging and Visualization Utilities for Cluster Scheduler.

Provides TensorBoard integration, CSV/history tracking, and automatic
reward curve plotting using Matplotlib.
"""

import os
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricLogger:
    """Tracks training statistics, writes TensorBoard logs, and renders reward curves."""

    def __init__(self, log_dir: str = "runs", run_name: Optional[str] = None):
        self.log_dir = log_dir
        if run_name:
            self.run_dir = os.path.join(log_dir, run_name)
        else:
            self.run_dir = log_dir
        os.makedirs(self.run_dir, exist_ok=True)

        self.writer = None
        if TENSORBOARD_AVAILABLE:
            try:
                self.writer = SummaryWriter(log_dir=self.run_dir)
            except Exception as e:
                print(f"[Logger] TensorBoard initialization warning: {e}")

        self.history: Dict[str, List[float]] = {
            "episode_reward": [],
            "ft_signal": [],
            "util_signal": [],
            "cost_signal": [],
            "actor_loss": [],
            "critic_loss": [],
            "noise_sigma": [],
        }

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar metric to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_episode(
        self,
        episode: int,
        reward: float,
        ft_signal: float,
        util_signal: float,
        cost_signal: float,
        actor_loss: float = 0.0,
        critic_loss: float = 0.0,
        noise_sigma: float = 0.0,
    ):
        """Record episode summary and write to TensorBoard."""
        self.history["episode_reward"].append(reward)
        self.history["ft_signal"].append(ft_signal)
        self.history["util_signal"].append(util_signal)
        self.history["cost_signal"].append(cost_signal)
        self.history["actor_loss"].append(actor_loss)
        self.history["critic_loss"].append(critic_loss)
        self.history["noise_sigma"].append(noise_sigma)

        if self.writer:
            self.writer.add_scalar("Reward/Total", reward, episode)
            self.writer.add_scalar("Reward/Fault_Tolerance", ft_signal, episode)
            self.writer.add_scalar("Reward/Utilization", util_signal, episode)
            self.writer.add_scalar("Reward/Cost", cost_signal, episode)
            self.writer.add_scalar("Loss/Actor", actor_loss, episode)
            self.writer.add_scalar("Loss/Critic", critic_loss, episode)
            self.writer.add_scalar("Exploration/Noise_Sigma", noise_sigma, episode)

    def save_reward_plot(self, filename: str = "reward_curve.png"):
        """Plot training reward curve with decomposed FT, Util, and Cost signals."""
        filepath = os.path.join(self.run_dir, filename)
        episodes = list(range(1, len(self.history["episode_reward"]) + 1))

        if len(episodes) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot 1: Total Reward and moving average
        ax1.plot(episodes, self.history["episode_reward"], label="Total Reward", color="blue", alpha=0.4)
        if len(episodes) >= 10:
            moving_avg = [
                sum(self.history["episode_reward"][max(0, i - 10) : i + 1])
                / len(self.history["episode_reward"][max(0, i - 10) : i + 1])
                for i in range(len(episodes))
            ]
            ax1.plot(episodes, moving_avg, label="Moving Avg (10 ep)", color="darkblue", linewidth=2)
        ax1.set_title("Training Reward Curve")
        ax1.set_ylabel("Reward")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        # Plot 2: Decomposed Reward Signals
        ax2.plot(episodes, self.history["ft_signal"], label="Fault Tolerance Signal", color="green", alpha=0.7)
        ax2.plot(episodes, self.history["util_signal"], label="Utilization Signal", color="orange", alpha=0.7)
        ax2.plot(episodes, self.history["cost_signal"], label="Cost Signal", color="purple", alpha=0.7)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Signal Value")
        ax2.set_title("Decomposed Underlying Signals (Unweighted)")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close(fig)
        return filepath

    def close(self):
        """Close TensorBoard writer and flush buffers."""
        if self.writer:
            self.writer.flush()
            self.writer.close()
