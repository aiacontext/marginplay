"""BRO-MARL: lift de BRO (Nauman et al. NeurIPS 2024) e TQC (Kuznetsov et al.
ICML 2020) para CTDE multi-agente (spec §4.5).

Cada agente carrega: actor + quantile critic + targets + 2 otimizadores Adam.
``BROMARLSystem`` agrupa os 5 agentes e o replay buffer, expondo ``act()``
(rollout) e ``train_step()`` (atualização CTDE com Huber-quantile loss).

Diferenças vs MADDPG vanilla (v1.0 da spec):
- 1 critic por agente (não twin) — LayerNorm regulariza Lipschitz, dispensa
  o ``min(Q_a, Q_b)`` patch do TD3/MATD3.
- Critic distributional: target é vetor de N_QUANTILES, loss é Huber-quantile.
  Scale-equivariant — agentes com reward em R$ bilhões não dominam o sistema.
- Target update Polyak τ=0.005 (mais conservador que MADDPG 0.01).
- Sem target policy smoothing, sem delayed actor — desnecessários quando o
  critic é regularizado por LN + quantile.

Gestão de memória MLX (spec §4.5; padrão Mini-Enedina):
- ``mx.eval`` granular após cada update (critic, actor, soft) — colapsa a
  janela de tracer graph de "5 agentes simultâneos" para "1 op por vez".
- ``mx.clear_cache`` por agente (não no fim do step) — devolve buffers
  transitórios ao pool antes de iniciar o próximo agente.
- ``_blend`` usa ``mlx.utils.tree_map`` em vez de recursão Python manual,
  eliminando alocação de dicts intermediários a cada Polyak.
- Cache MLX MANTIDO ATIVO (set_cache_limit não é chamado): o cache é a
  defesa primária do allocator contra o Resource limit (499000) — desativá-lo
  força nova alocação Metal a cada operação, piora o problema.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from agents.buffer import ReplayBuffer
from agents.definitions import AgentSpec, all_specs, total_act_dim
from agents.networks import (
    Actor,
    QuantileCritic,
    huber_quantile_loss,
    quantile_taus,
)
from agents.noise import OUNoise
from core.environment import AGENTS, MarginPlayEnv
from core.world import Actions

GAMMA_DEFAULT = 0.95
TAU_DEFAULT = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
N_QUANTILES = 25
HUBER_KAPPA = 1.0


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: θ_target ← τ*θ_source + (1-τ)*θ_target.

    Usa ``mlx.utils.tree_map`` para evitar alocação de dicts Python aninhados
    a cada chamada — em treinos longos isso era a maior fonte de buffers
    transitórios (5 agentes × ~20 layers × 2 redes target × 1500 steps/ep).
    """
    blended = tree_map(
        lambda t, s: tau * s + (1.0 - tau) * t,
        target.parameters(),
        source.parameters(),
    )
    target.update(blended)


@dataclass
class BROMARLAgent:
    """Per-agent: actor + quantile critic + targets + 2 otimizadores Adam."""

    spec: AgentSpec
    actor: Actor
    critic: QuantileCritic
    target_actor: Actor
    target_critic: QuantileCritic
    actor_opt: optim.Adam
    critic_opt: optim.Adam
    noise: OUNoise

    @classmethod
    def build(
        cls,
        spec: AgentSpec,
        state_dim: int,
        total_act_dim_value: int,
        n_quantiles: int = N_QUANTILES,
        seed: int = 0,
    ) -> BROMARLAgent:
        """Constrói actor + critic + targets + otimizadores."""
        actor = Actor(spec.obs_dim, spec.act_dim, output_activation=spec.output_activation)
        critic = QuantileCritic(state_dim, total_act_dim_value, n_quantiles=n_quantiles)
        target_actor = Actor(spec.obs_dim, spec.act_dim, output_activation=spec.output_activation)
        target_critic = QuantileCritic(state_dim, total_act_dim_value, n_quantiles=n_quantiles)
        target_actor.update(actor.parameters())
        target_critic.update(critic.parameters())
        return cls(
            spec=spec,
            actor=actor,
            critic=critic,
            target_actor=target_actor,
            target_critic=target_critic,
            actor_opt=optim.Adam(learning_rate=ACTOR_LR),
            critic_opt=optim.Adam(learning_rate=CRITIC_LR),
            noise=OUNoise(spec.act_dim, seed=seed),
        )

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """Seleciona ação (numpy) com exploração OU opcional."""
        obs_mx = mx.array(obs.astype(np.float32))[None, :]
        action_raw = np.asarray(self.actor(obs_mx)[0])
        if explore:
            noise = self.noise.sample()
            if self.spec.output_activation == "softmax":
                # Para alocação: perturbar e renormalizar (mantém simplex).
                perturbed = np.maximum(action_raw + 0.1 * noise, 1e-3)
                action_raw = perturbed / perturbed.sum()
            else:
                action_raw = np.tanh(np.arctanh(np.clip(action_raw, -0.99, 0.99)) + noise)
        return self.spec.rescale_action(action_raw)


@dataclass
class BROMARLSystem:
    """Agrupa os 5 agentes + buffer + interface CTDE de treino."""

    agents: dict[str, BROMARLAgent]
    buffer: ReplayBuffer
    state_dim: int
    total_act_dim: int
    n_quantiles: int

    @classmethod
    def build(
        cls,
        state_dim: int,
        buffer_capacity: int = 1_000_000,
        n_quantiles: int = N_QUANTILES,
    ) -> BROMARLSystem:
        tad = total_act_dim()
        return cls(
            agents={
                spec.id: BROMARLAgent.build(
                    spec, state_dim, tad, n_quantiles=n_quantiles, seed=hash(spec.id) & 0xFFFF
                )
                for spec in all_specs()
            },
            buffer=ReplayBuffer(capacity=buffer_capacity),
            state_dim=state_dim,
            total_act_dim=tad,
            n_quantiles=n_quantiles,
        )

    # ------------------------------------------------------------------
    # Inferência (rollout)
    # ------------------------------------------------------------------
    def act(self, obs: dict[str, np.ndarray], explore: bool = True) -> Actions:
        return {ag: self.agents[ag].act(obs[ag], explore=explore) for ag in AGENTS}

    # ------------------------------------------------------------------
    # Treino (CTDE com Huber-quantile loss)
    # ------------------------------------------------------------------
    def train_step(
        self,
        batch_size: int = 1024,
        gamma: float = GAMMA_DEFAULT,
        tau: float = TAU_DEFAULT,
    ) -> dict[str, dict[str, float]]:
        """Um passo de treino para todos os agentes. Retorna dict de métricas."""
        if len(self.buffer) < batch_size:
            return {}
        batch = self.buffer.sample(batch_size, AGENTS)
        losses: dict[str, dict[str, float]] = {}
        taus = quantile_taus(self.n_quantiles)

        # Pré-computa ações-alvo (target_actor) e ações observadas (concat).
        next_actions_per_agent = {
            ag: self.agents[ag].target_actor(batch.next_obs[ag]) for ag in AGENTS
        }
        next_actions_concat = mx.concatenate([next_actions_per_agent[ag] for ag in AGENTS], axis=-1)
        all_actions_concat = mx.concatenate([batch.actions[ag] for ag in AGENTS], axis=-1)

        for ag in AGENTS:
            agent = self.agents[ag]
            losses[ag] = {}

            # ---- Critic distributional: Huber-quantile sobre Z_target ----
            z_next = agent.target_critic(batch.next_states, next_actions_concat)
            r = batch.rewards[ag][:, None]
            d = batch.dones[:, None]
            q_target = r + gamma * (1.0 - d) * z_next
            q_target = mx.stop_gradient(q_target)

            def critic_loss_fn(params, q_tgt=q_target, _agent=agent):
                _agent.critic.update(params)
                q_pred = _agent.critic(batch.states, all_actions_concat)
                return huber_quantile_loss(q_pred, q_tgt, taus, kappa=HUBER_KAPPA)

            critic_loss, critic_grads = mx.value_and_grad(critic_loss_fn)(agent.critic.parameters())
            agent.critic_opt.update(agent.critic, critic_grads)
            # Eval granular após o critic update — materializa params + Adam
            # state imediatamente, fechando a janela de tracer graph desse
            # update antes de começar o próximo (padrão Mini-Enedina).
            mx.eval(agent.critic.parameters(), agent.critic_opt.state)

            losses[ag]["critic"] = float(critic_loss.item())
            losses[ag]["q_target_mean"] = float(mx.mean(q_target).item())
            losses[ag]["q_target_max"] = float(mx.max(mx.abs(q_target)).item())
            losses[ag]["reward_mean"] = float(mx.mean(batch.rewards[ag]).item())

            # ---- Actor: ascende em E[Z(s, μ_i(o_i), a_-i)] = mean(quantis) ----
            def actor_loss_fn(params, ag_id=ag, _agent=agent):
                _agent.actor.update(params)
                my_action = _agent.actor(batch.obs[ag_id])
                pieces: list[mx.array] = []
                for other in AGENTS:
                    pieces.append(my_action if other == ag_id else batch.actions[other])
                composed = mx.concatenate(pieces, axis=-1)
                z = _agent.critic(batch.states, composed)
                return -mx.mean(z)

            actor_loss, actor_grads = mx.value_and_grad(actor_loss_fn)(agent.actor.parameters())
            agent.actor_opt.update(agent.actor, actor_grads)
            mx.eval(agent.actor.parameters(), agent.actor_opt.state)
            losses[ag]["actor"] = float(actor_loss.item())

            # ---- Soft update Polyak (sem twin, sem delayed update) ----
            _soft_update(agent.target_actor, agent.actor, tau)
            _soft_update(agent.target_critic, agent.critic, tau)
            mx.eval(agent.target_actor.parameters(), agent.target_critic.parameters())

            # Devolve buffers transitórios desse agente ao pool antes de
            # iniciar o próximo. Essencial para evitar Resource limit (499000)
            # do Metal allocator em treinos longos.
            mx.clear_cache()

        return losses


def encode_state(env: MarginPlayEnv) -> np.ndarray:
    """Encode WorldState como vetor flat para o Critic centralizado.

    Layout (22 dims): 6 escalares + 4 stocks zonais × 4 zonas.
    """
    s = env.world.state  # type: ignore[union-attr]
    n_total = max(s.N_total, 1e-9)
    n_rel = s.N / n_total
    return np.concatenate(
        [
            np.array(
                [
                    s.R / max(env.scenario.URR_Gbbl, 1e-9),
                    s.E_amb,
                    s.P_efetiva,
                    s.preco / 100.0,
                    s.roy_periodo / 1e9,
                    s.gini,
                ],
                dtype=np.float32,
            ),
            s.K_pub.astype(np.float32),
            s.K_hum.astype(np.float32),
            n_rel.astype(np.float32),
            s.C_inst.astype(np.float32),
        ]
    )
