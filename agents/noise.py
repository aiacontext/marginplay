"""Ruído Ornstein-Uhlenbeck para exploração contínua (spec §4.6)."""

from __future__ import annotations

import numpy as np


class OUNoise:
    """Processo Ornstein-Uhlenbeck temporalmente correlacionado.

    Parâmetros típicos para MADDPG: μ=0, θ=0.15, σ=0.2 (decay 0.9995/episódio).
    """

    def __init__(
        self,
        dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)
        self.state = np.full(dim, mu, dtype=float)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * self._rng.standard_normal(self.dim)
        self.state = self.state + dx
        return self.state.copy()

    def reset(self) -> None:
        self.state = np.full(self.dim, self.mu, dtype=float)

    def decay(self, factor: float = 0.9995) -> None:
        """Reduz σ multiplicativamente (chamado a cada episódio)."""
        self.sigma *= factor
