"""Actor e QuantileCritic em MLX para BRO-MARL (spec §4.1, §4.3).

CTDE: Actor é local (vê só obs do agente), Critic é centralizado distributional
(vê estado global + ações de todos, devolve N_QUANTILES estimativas).

Diferenças em relação à v1.0 (MADDPG vanilla):
- LayerNorm após cada hidden em Actor e Critic (regulariza Lipschitz; substitui
  twin-min como mecanismo anti-overestimation — RLPD ICML 2023, ReBRAC NeurIPS
  2023, Nauman et al. NeurIPS 2024).
- Critic distributional (TQC, Kuznetsov ICML 2020): saída são N_QUANTILES
  estimativas dos quantis do retorno, não uma média escalar. Quantile
  regression é scale-equivariant — agentes com reward em R$ bilhões não geram
  gradients ordens de magnitude maiores que agentes em escala unitária.
- Critic mais largo (512×512) — BRO mostra que escalar critic ajuda mais que
  escalar actor; em CTDE o critic recebe o input mais rico.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

N_QUANTILES_DEFAULT = 25


class Actor(nn.Module):
    """Actor local: obs -> action. LayerNorm após cada Linear hidden.

    Arquitetura: Linear(hidden) -> LN -> ReLU -> Linear(hidden//2) -> LN -> ReLU
    -> Linear(act_dim) -> Tanh ou Softmax.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: int = 128,
        output_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if output_activation not in ("tanh", "softmax"):
            raise ValueError(f"output_activation desconhecida: {output_activation}")
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.ln2 = nn.LayerNorm(hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, act_dim)
        self.output_activation = output_activation

    def __call__(self, obs: mx.array) -> mx.array:
        x = nn.relu(self.ln1(self.fc1(obs)))
        x = nn.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        if self.output_activation == "softmax":
            return mx.softmax(x, axis=-1)
        return mx.tanh(x)


class QuantileCritic(nn.Module):
    """Critic centralizado distributional: (state, all_actions) -> N quantis.

    Devolve tensor (B, n_quantiles) — estimativa de Z(s,a) nos níveis
    τ_k = (k+0.5)/N. E[Z] = mean(...) é o equivalente do Q escalar do MADDPG.
    """

    def __init__(
        self,
        state_dim: int,
        total_act_dim: int,
        hidden: int = 512,
        n_quantiles: int = N_QUANTILES_DEFAULT,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.fc1 = nn.Linear(state_dim + total_act_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, n_quantiles)

    def __call__(self, state: mx.array, actions: mx.array) -> mx.array:
        x = mx.concatenate([state, actions], axis=-1)
        x = nn.relu(self.ln1(self.fc1(x)))
        x = nn.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

    def expectation(self, state: mx.array, actions: mx.array) -> mx.array:
        """E[Z(s,a)] = média dos quantis. Usado pelo gradiente do actor."""
        return mx.mean(self.__call__(state, actions), axis=-1)


def quantile_taus(n_quantiles: int) -> mx.array:
    """Níveis dos quantis: τ_k = (k + 0.5) / N, k = 0..N-1."""
    return (mx.arange(n_quantiles, dtype=mx.float32) + 0.5) / n_quantiles


def huber_quantile_loss(
    q_pred: mx.array,
    q_target: mx.array,
    taus: mx.array,
    kappa: float = 1.0,
) -> mx.array:
    """Loss Huber-quantile (TQC, Kuznetsov ICML 2020).

    q_pred:   (B, N) — quantis preditos.
    q_target: (B, N) — quantis-alvo (atomic, B targets cada um replicado).
    taus:     (N,)   — níveis dos quantis preditos.

    Pairwise: para cada par (i, j), erro = q_target_j - q_pred_i, ponderado
    por |τ_i − 1{erro<0}| via Huber(κ). Retorna escalar.
    """
    pred = q_pred[:, :, None]  # (B, N, 1)
    tgt = q_target[:, None, :]  # (B, 1, N)
    diff = tgt - pred

    abs_diff = mx.abs(diff)
    huber = mx.where(
        abs_diff <= kappa,
        0.5 * diff * diff,
        kappa * (abs_diff - 0.5 * kappa),
    )
    indicator = mx.where(diff < 0, mx.array(1.0), mx.array(0.0))
    weight = mx.abs(taus[None, :, None] - indicator)
    loss_per_pair = weight * huber / kappa
    return mx.mean(mx.sum(mx.mean(loss_per_pair, axis=2), axis=1))
