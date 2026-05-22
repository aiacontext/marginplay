"""Experience replay buffer multi-agente (spec §4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class Transition:
    """Uma transição armazenada no buffer (formato multi-agente)."""

    state: np.ndarray
    """Estado global concatenado (para o Critic centralizado)."""

    obs: dict[str, np.ndarray]
    """Observação local de cada agente."""

    actions: dict[str, np.ndarray]
    """Ação tomada por cada agente."""

    rewards: dict[str, float]
    """Recompensa recebida por cada agente."""

    next_state: np.ndarray
    next_obs: dict[str, np.ndarray]
    done: bool


@dataclass
class Batch:
    """Batch amostrado do replay buffer, já em mx.array por agente."""

    states: mx.array  # (B, state_dim)
    obs: dict[str, mx.array]  # cada (B, obs_dim_i)
    actions: dict[str, mx.array]  # cada (B, act_dim_i)
    rewards: dict[str, mx.array]  # cada (B,)
    next_states: mx.array  # (B, state_dim)
    next_obs: dict[str, mx.array]
    dones: mx.array  # (B,)


class ReplayBuffer:
    """Buffer circular de transições multi-agente.

    Capacidade default 1M (spec §4.7); cada slot guarda obs/actions de
    todos os agentes simultaneamente para preservar a temporalidade conjunta.
    """

    def __init__(self, capacity: int = 1_000_000, seed: int | None = None) -> None:
        self.capacity = capacity
        self._buffer: list[Transition] = []
        self._position = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, transition: Transition) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int, agent_ids: tuple[str, ...]) -> Batch:
        """Amostra ``batch_size`` transições e devolve em ``Batch`` mx.arrays."""
        if batch_size > len(self._buffer):
            raise ValueError(f"batch_size {batch_size} > buffer size {len(self._buffer)}")
        idxs = self._rng.choice(len(self._buffer), batch_size, replace=False)
        items = [self._buffer[i] for i in idxs]

        states = mx.array(np.stack([t.state for t in items]))
        next_states = mx.array(np.stack([t.next_state for t in items]))
        dones = mx.array(np.array([float(t.done) for t in items]))

        obs: dict[str, mx.array] = {}
        next_obs: dict[str, mx.array] = {}
        actions: dict[str, mx.array] = {}
        rewards: dict[str, mx.array] = {}
        for ag in agent_ids:
            obs[ag] = mx.array(np.stack([t.obs[ag] for t in items]).astype(np.float32))
            next_obs[ag] = mx.array(np.stack([t.next_obs[ag] for t in items]).astype(np.float32))
            actions[ag] = mx.array(np.stack([t.actions[ag] for t in items]).astype(np.float32))
            rewards[ag] = mx.array(np.array([t.rewards[ag] for t in items]).astype(np.float32))

        return Batch(
            states=states.astype(mx.float32),
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_states=next_states.astype(mx.float32),
            next_obs=next_obs,
            dones=dones.astype(mx.float32),
        )
