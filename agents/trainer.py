"""Loop de treino BRO-MARL (spec §4.5, §4.7).

Episódio = 15 steps (horizonte 30 anos). Por episódio:
1. ``env.reset()``.
2. Rollout: action -> step -> push transition.
3. Após cada step (uma vez buffer atinge ``warmup_steps``): train_step.
4. Decay OU noise.
5. Log de retorno por agente + W médio.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from agents.bro_marl import BROMARLSystem, encode_state
from agents.buffer import Transition
from core.environment import AGENTS, MarginPlayEnv

EpisodeCallback = Callable[[int, "EpisodeStats", BROMARLSystem], None]

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Hiperparâmetros do treino (spec §4.7 com escala reduzida para smoke)."""

    n_episodes: int = 1000
    batch_size: int = 256
    warmup_episodes: int = 5
    """Acumula transições por N episódios antes do primeiro train_step."""

    train_every_steps: int = 1
    """Quantos env steps por train_step."""

    log_every_episodes: int = 25
    seed: int = 0
    noise_decay: float = 0.999
    """Multiplicador do σ do OU noise por episódio."""


@dataclass
class EpisodeStats:
    """Métricas agregadas de um episódio."""

    episode: int
    returns: dict[str, float]
    final_W: float  # noqa: N815 -- nome matemático da spec
    final_E_amb: float  # noqa: N815 -- nome matemático da spec
    final_R: float  # noqa: N815 -- nome matemático da spec
    losses_actor: dict[str, float] = field(default_factory=dict)
    losses_critic: dict[str, float] = field(default_factory=dict)
    # Diagnósticos para investigar bootstrap divergence (média no episódio).
    q_target_mean: dict[str, float] = field(default_factory=dict)
    q_target_max: dict[str, float] = field(default_factory=dict)
    critic_grad_norm: dict[str, float] = field(default_factory=dict)
    actor_grad_norm: dict[str, float] = field(default_factory=dict)
    reward_mean_batch: dict[str, float] = field(default_factory=dict)


class Trainer:
    """Orquestra o treino BRO-MARL num único env (extensível para vec env)."""

    def __init__(
        self,
        env: MarginPlayEnv,
        system: BROMARLSystem,
        config: TrainerConfig | None = None,
        on_episode_end: EpisodeCallback | None = None,
    ) -> None:
        self.env = env
        self.system = system
        self.config = config or TrainerConfig()
        self.on_episode_end = on_episode_end
        """Callback opcional disparado após cada episódio (uso típico:
        checkpoints periódicos, métricas externas)."""
        self._returns_history: deque[dict[str, float]] = deque(maxlen=100)

    def run(self) -> list[EpisodeStats]:
        """Loop principal de treino."""
        cfg = self.config
        all_stats: list[EpisodeStats] = []

        for ep in range(cfg.n_episodes):
            stats = self._run_episode(ep)
            all_stats.append(stats)
            if (ep + 1) % cfg.log_every_episodes == 0:
                self._log_block(ep + 1, all_stats[-cfg.log_every_episodes :])
            # Decay do ruído após cada episódio
            for agent in self.system.agents.values():
                agent.noise.decay(cfg.noise_decay)
            if self.on_episode_end is not None:
                self.on_episode_end(ep, stats, self.system)
        return all_stats

    def _run_episode(self, episode_idx: int) -> EpisodeStats:
        cfg = self.config
        obs = self.env.reset()
        prev_state = encode_state(self.env)
        cum_returns: dict[str, float] = {ag: 0.0 for ag in AGENTS}
        diag_keys = (
            "critic",
            "actor",
            "critic_grad_norm",
            "actor_grad_norm",
            "q_target_mean",
            "q_target_max",
            "reward_mean",
        )
        diag_history: dict[str, dict[str, list[float]]] = {
            ag: {k: [] for k in diag_keys} for ag in AGENTS
        }
        last_info = None

        for _ in range(self.env.world.horizon_steps):  # type: ignore[union-attr]
            actions = self.system.act(obs, explore=True)
            result = self.env.step(actions)
            next_state = encode_state(self.env)

            self.system.buffer.push(
                Transition(
                    state=prev_state,
                    obs=obs,
                    actions=actions,
                    rewards=result.rewards,
                    next_state=next_state,
                    next_obs=result.observations,
                    done=result.done,
                )
            )
            for ag, r in result.rewards.items():
                cum_returns[ag] += r

            obs = result.observations
            prev_state = next_state
            last_info = result.info

            # Train step (após warmup + buffer suficiente)
            if episode_idx >= cfg.warmup_episodes and len(self.system.buffer) >= cfg.batch_size:
                losses = self.system.train_step(batch_size=cfg.batch_size)
                if losses:
                    for ag, ls in losses.items():
                        for k in diag_keys:
                            if k in ls:
                                diag_history[ag][k].append(ls[k])

            if result.done:
                break

        self._returns_history.append(cum_returns)

        def _mean_per_agent(key: str) -> dict[str, float]:
            return {
                ag: float(np.mean(diag_history[ag][key])) if diag_history[ag][key] else float("nan")
                for ag in AGENTS
            }

        def _max_per_agent(key: str) -> dict[str, float]:
            return {
                ag: float(np.max(diag_history[ag][key])) if diag_history[ag][key] else float("nan")
                for ag in AGENTS
            }

        return EpisodeStats(
            episode=episode_idx,
            returns=cum_returns,
            final_W=last_info.W if last_info else 0.0,
            final_E_amb=self.env.world.state.E_amb,  # type: ignore[union-attr]
            final_R=self.env.world.state.R,  # type: ignore[union-attr]
            losses_critic=_mean_per_agent("critic"),
            losses_actor=_mean_per_agent("actor"),
            q_target_mean=_mean_per_agent("q_target_mean"),
            q_target_max=_max_per_agent("q_target_max"),
            critic_grad_norm=_mean_per_agent("critic_grad_norm"),
            actor_grad_norm=_mean_per_agent("actor_grad_norm"),
            reward_mean_batch=_mean_per_agent("reward_mean"),
        )

    def _log_block(self, ep: int, recent: list[EpisodeStats]) -> None:
        avg_returns = {ag: float(np.mean([s.returns[ag] for s in recent])) for ag in AGENTS}
        avg_w = float(np.mean([s.final_W for s in recent]))
        avg_e = float(np.mean([s.final_E_amb for s in recent]))
        avg_r = float(np.mean([s.final_R for s in recent]))
        last = recent[-1]
        msg_returns = " ".join(f"{ag.split('_')[0]}={avg_returns[ag]:+.2f}" for ag in AGENTS)
        msg_critic = " ".join(
            f"{ag.split('_')[0]}={last.losses_critic.get(ag, float('nan')):.3f}" for ag in AGENTS
        )
        msg_qtgt = " ".join(
            f"{ag.split('_')[0]}={last.q_target_mean.get(ag, float('nan')):+.2f}/"
            f"{last.q_target_max.get(ag, float('nan')):.2f}"
            for ag in AGENTS
        )
        logger.info(
            "ep=%4d | W=%.3f E_amb=%.3f R=%.2f | returns: %s | critic_loss: %s",
            ep,
            avg_w,
            avg_e,
            avg_r,
            msg_returns,
            msg_critic,
        )
        logger.info("         q_tgt(mean/|max|): %s", msg_qtgt)
