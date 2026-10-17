"""
Action Exploration Noise for MADDPG.

Supports Gaussian noise with linear decay and Ornstein-Uhlenbeck (OU) process noise.
"""

from typing import Optional, Tuple
import numpy as np


class ActionNoise:
    """Base class for action exploration noise."""

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        raise NotImplementedError

    def step(self):
        pass

    def reset(self):
        pass

    def get_sigma(self) -> float:
        raise NotImplementedError


class GaussianNoise(ActionNoise):
    """Gaussian noise with linear decay over training steps."""

    def __init__(
        self,
        sigma_start: float = 0.3,
        sigma_end: float = 0.05,
        decay_steps: int = 50000,
    ):
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = max(1, decay_steps)
        self.current_step = 0
        self.sigma = sigma_start

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        return np.random.normal(0.0, self.sigma, size=shape).astype(np.float32)

    def step(self):
        self.current_step += 1
        fraction = min(1.0, self.current_step / self.decay_steps)
        self.sigma = self.sigma_start - fraction * (self.sigma_start - self.sigma_end)

    def reset(self):
        pass

    def get_sigma(self) -> float:
        return float(self.sigma)


class OrnsteinUhlenbeckNoise(ActionNoise):
    """Ornstein-Uhlenbeck process noise with mean-reversion and decaying sigma."""

    def __init__(
        self,
        theta: float = 0.15,
        sigma_start: float = 0.2,
        sigma_end: float = 0.05,
        decay_steps: int = 50000,
        mu: float = 0.0,
        dt: float = 1e-2,
    ):
        self.theta = theta
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.decay_steps = max(1, decay_steps)
        self.mu = mu
        self.dt = dt
        self.current_step = 0
        self.sigma = sigma_start
        self.state: Optional[np.ndarray] = None

    def sample(self, shape: Tuple[int, ...]) -> np.ndarray:
        if self.state is None or self.state.shape != shape:
            self.state = np.ones(shape, dtype=np.float32) * self.mu

        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=shape).astype(np.float32)
        )
        self.state = self.state + dx
        return self.state.copy()

    def step(self):
        self.current_step += 1
        fraction = min(1.0, self.current_step / self.decay_steps)
        self.sigma = self.sigma_start - fraction * (self.sigma_start - self.sigma_end)

    def reset(self):
        self.state = None

    def get_sigma(self) -> float:
        return float(self.sigma)


def create_noise(
    noise_type: str = "gaussian",
    sigma_start: float = 0.3,
    sigma_end: float = 0.05,
    decay_steps: int = 50000,
    ou_theta: float = 0.15,
    ou_sigma: float = 0.2,
) -> ActionNoise:
    """Factory function for action noise."""
    if noise_type.lower() == "ou":
        return OrnsteinUhlenbeckNoise(
            theta=ou_theta,
            sigma_start=ou_sigma,
            sigma_end=sigma_end,
            decay_steps=decay_steps,
        )
    return GaussianNoise(
        sigma_start=sigma_start,
        sigma_end=sigma_end,
        decay_steps=decay_steps,
    )
