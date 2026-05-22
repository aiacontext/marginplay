"""Definições por agente: dimensões, ativações de saída, ranges.

Centraliza o que cada agente espera/produz, derivado da spec §3.2-§3.6 e
§4.2 (estendido para zonas em §2.6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.environment import (
    ACT_DIMS,
    AGENT_ANP,
    AGENT_COM,
    AGENT_FED,
    AGENT_GOV,
    AGENT_IBAMA,
    AGENT_OPER,
    AGENTS,
    OBS_DIMS,
)


@dataclass(frozen=True)
class AgentSpec:
    """Especificação de um agente do jogo."""

    id: str
    obs_dim: int
    act_dim: int
    output_activation: str
    """``"softmax"`` para alocações orçamentárias, ``"tanh"`` para escalares."""

    act_low: np.ndarray
    """Limite inferior de cada componente da ação (após rescaling)."""

    act_high: np.ndarray
    """Limite superior de cada componente da ação (após rescaling)."""

    def rescale_action(self, raw: np.ndarray) -> np.ndarray:
        """Rescala saída do Actor (tanh ∈ [-1,1] ou softmax ∈ simplex) para o
        intervalo válido ``[act_low, act_high]``.

        - Softmax: já em [0,1] e somando 1 — preserva (sem rescaling).
        - Tanh: mapeia [-1,1] → [act_low, act_high].
        """
        if self.output_activation == "softmax":
            return np.asarray(raw, dtype=float)
        # tanh
        raw = np.asarray(raw, dtype=float)
        scaled = self.act_low + (raw + 1.0) * 0.5 * (self.act_high - self.act_low)
        return np.clip(scaled, self.act_low, self.act_high)


# Construção dos AgentSpec baseada na spec §3
SPECS: dict[str, AgentSpec] = {
    AGENT_GOV: AgentSpec(
        id=AGENT_GOV,
        obs_dim=OBS_DIMS[AGENT_GOV],
        act_dim=ACT_DIMS[AGENT_GOV],
        output_activation="softmax",  # 6 alocações sobre RCL livre, somam 1
        act_low=np.zeros(ACT_DIMS[AGENT_GOV]),
        act_high=np.ones(ACT_DIMS[AGENT_GOV]),
    ),
    AGENT_OPER: AgentSpec(
        id=AGENT_OPER,
        obs_dim=OBS_DIMS[AGENT_OPER],
        act_dim=ACT_DIMS[AGENT_OPER],
        output_activation="tanh",
        # alpha_invest ∈ [0.5, 1.2], alpha_seg ∈ [0, 1]
        act_low=np.array([0.5, 0.0]),
        act_high=np.array([1.2, 1.0]),
    ),
    AGENT_ANP: AgentSpec(
        id=AGENT_ANP,
        obs_dim=OBS_DIMS[AGENT_ANP],
        act_dim=ACT_DIMS[AGENT_ANP],
        output_activation="tanh",
        # ritmo_aprov_PD ∈ [0,1] — velocidade aprovação Plano de Desenvolvimento
        # rigor_seg_op ∈ [0,1] — exigência de segurança operacional (Res. 882/2022)
        act_low=np.array([0.0, 0.0]),
        act_high=np.array([1.0, 1.0]),
    ),
    AGENT_IBAMA: AgentSpec(
        id=AGENT_IBAMA,
        obs_dim=OBS_DIMS[AGENT_IBAMA],
        act_dim=ACT_DIMS[AGENT_IBAMA],
        output_activation="tanh",
        # phi_fisc_amb ∈ [0,1], exigencia_compensacao ∈ [0,1]
        act_low=np.array([0.0, 0.0]),
        act_high=np.array([1.0, 1.0]),
    ),
    AGENT_COM: AgentSpec(
        id=AGENT_COM,
        obs_dim=OBS_DIMS[AGENT_COM],
        act_dim=ACT_DIMS[AGENT_COM],
        output_activation="tanh",
        # mobilizacao ∈ [0,1], preferencia_amb ∈ [0,1]
        act_low=np.array([0.0, 0.0]),
        act_high=np.array([1.0, 1.0]),
    ),
    AGENT_FED: AgentSpec(
        id=AGENT_FED,
        obs_dim=OBS_DIMS[AGENT_FED],
        act_dim=ACT_DIMS[AGENT_FED],
        output_activation="tanh",
        # v2: ritmo_leiloes ∈ [0,1], alpha_cide ∈ [0,1], fisc_amb ∈ [0,1]
        # (substitui frac_repasse — divisão é estatutária, não discricionária)
        act_low=np.array([0.0, 0.0, 0.0]),
        act_high=np.array([1.0, 1.0, 1.0]),
    ),
}


def all_specs() -> list[AgentSpec]:
    """Retorna AgentSpecs na ordem canônica de AGENTS."""
    return [SPECS[a] for a in AGENTS]


def state_dim_global(extra_scalars: int = 6, n_zones: int = 4) -> int:
    """Dimensão do estado global passado ao Critic centralizado.

    Spec §4.2 (extendido): 6 escalares (R, E_amb, P_efetiva, Preço, Roy, Gini)
    + 4 zonas × 4 stocks zonais (K_pub, K_hum, N_rel, C_inst) = 22 default.
    """
    return extra_scalars + 4 * n_zones


def total_act_dim() -> int:
    """Soma das dimensões de ação de todos os agentes."""
    return sum(s.act_dim for s in all_specs())
