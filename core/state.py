"""Estado do mundo (spec §2.2): stocks zonais + escalares.

Cada instância de ``World`` simula UMA UF da Margem Equatorial. Stocks
sociais (K_pub, K_hum, N, C_inst) são vetores indexados por zona (§2.6);
stocks físicos/macro (R, E_amb, Preço) são escalares no nível UF.

Carregamento de condições iniciais a partir de
``data/processed/zonas.parquet`` (gerado pelo pipeline bulk_calibration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Ordem canônica das zonas — a posição no vetor é semanticamente significativa
ZONES: tuple[str, ...] = (
    "costa_produtora",
    "costa_nao_produtora",
    "interior_medio_alto",
    "interior_baixo",
)
N_ZONES = len(ZONES)


@dataclass
class WorldState:
    """Estado do mundo num instante ``t``.

    Convenção de unidades:
    - ``t``: índice de step (0..15), cada step = 2 anos (spec §2.1)
    - ``R``: reservas em Gbbl (giga-barris)
    - ``preco``: USD/bbl Brent
    - ``E_amb``: índice [0, 1] (0 pristino, 1 degradação máxima)
    - ``P_efetiva``: produção corrente em Mbbl/d
    - ``roy_periodo``: royalties recebidos no período em R$ (escala UF)
    - ``gini``: índice de Gini intra-UF [0, 1]
    - Vetores zonais (shape (4,)):
      - ``K_pub``, ``K_hum``, ``C_inst``: índices [0, 1]
      - ``N``: população em milhares de habitantes
    """

    # Tempo
    t: int = 0

    # Escalares físicos / macro
    R: float = 0.0
    E_amb: float = 0.05
    P_efetiva: float = 0.0
    preco: float = 0.0
    roy_periodo: float = 0.0
    gini: float = 0.0
    W: float = 0.0  # bem-estar social agregado
    fundo_soberano: float = 0.0  # stock R$ acumulado pelo gov_estadual (spec §2.3.3)

    # ----- Capital privado e PIB estadual (spec §2.3.4 — Cobb-Douglas) -----
    K_priv: float = 0.0  # capital privado estadual em R$ bilhões
    pib_estadual: float = 0.0  # PIB estadual em R$ bilhões (calculado endogenamente)

    # ----- Receitas do estado (spec §2.3.5 — RCL decomposta) -----
    icms_periodo: float = 0.0  # endógeno ao PIB (elasticidade ε~1.10)
    fpe_periodo: float = 0.0  # cota MA fixa (LC 143/2013, ~7.2%)
    fundeb_periodo: float = 0.0  # complementação federal líquida (EC 108/2020)
    outras_receitas: float = 0.0  # IPVA + ITCMD + taxas + SUS (drift exógeno)

    # Vetores zonais — populados em load_initial_state
    K_pub: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))
    K_hum: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))
    K_saude: np.ndarray = field(
        default_factory=lambda: np.zeros(N_ZONES)
    )  # spec §2.3.6 (Grossman 1972 + WHO Building Blocks)
    N: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))
    C_inst: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))

    # Estoque de pressão territorial acumulada (H-TERR-2, calibração CPT/INCRA/CIMI 2024).
    # Captura o caráter histerético dos conflitos territoriais documentado em
    # Almeida (NAEA/UFPA 2008) e Escobar 2008: pressão acumulada não retorna ao
    # zero quando a degradação cessa. Atualização: S_{t+1} = (1-δ)·S_t + λ·max(0, ΔE_amb + ω·R_extr).
    # Valor inicial calibrado de CPT 2020-2023 (~200 conflitos/ano MA → S₀ ≈ 0.4 zonal médio).
    S_pressao_terr: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))

    # Buffer de investimentos em educação com defasagem (spec §2.2.3, τ=4 steps = 8 anos)
    I_educ_lag: np.ndarray = field(default_factory=lambda: np.zeros((N_ZONES, 4)))
    # Buffer de investimentos em saúde com defasagem menor (τ=2 steps = 4 anos, Grossman)
    I_saude_lag: np.ndarray = field(default_factory=lambda: np.zeros((N_ZONES, 2)))

    @property
    def N_total(self) -> float:  # noqa: N802 -- nome matemático da spec
        """População total da UF em milhares."""
        return float(self.N.sum())

    @property
    def ano_corrente(self) -> int:
        """Ano absoluto correspondente ao step atual (spec §2.1: 2028 = step 0)."""
        return 2028 + 2 * self.t

    def avg_pop(self, x: np.ndarray) -> float:
        """Média ponderada por população (spec §2.6)."""
        n = self.N_total
        if n <= 0:
            return float(np.nan)
        return float(np.dot(x, self.N) / n)


def load_initial_state(
    uf: str = "MA",
    zonas_parquet: Path | None = None,
    R0_Gbbl: float = 8.0,  # noqa: N803 -- nome matemático da spec
    preco0_usd: float = 80.0,
    c_inst_inicial_multiplier: float = 1.0,
) -> WorldState:
    """Carrega WorldState inicial para uma UF a partir do calibration parquet.

    Args:
        uf: sigla da UF (``"MA"``, ``"AP"``, ``"PA"``, ``"RN"``).
        zonas_parquet: path do parquet zonal; default
            ``data/processed/zonas.parquet``.
        R0_Gbbl: reservas iniciais em Gbbl (cenário URR P50 = 8 por padrão).
        preco0_usd: preço inicial Brent USD/bbl (~80 = média 2020-2026 EIA).

    Returns:
        WorldState populado com dados reais da UF.
    """
    path = zonas_parquet or Path("data/processed/zonas.parquet")
    df = pd.read_parquet(path)
    df_uf = df[df["uf"] == uf].set_index("zona")
    df_uf = df_uf.reindex(list(ZONES))  # garante ordem das zonas

    n_pop_milhares = (df_uf["populacao"].to_numpy() / 1000.0).astype(float)
    k_hum_arr = df_uf["k_hum"].to_numpy().astype(float)
    # K_saude proxy: correlato a IDH-Longevidade. Como não temos a dimensão
    # isolada, usamos K_hum × 0.95 (educação e saúde são complementares na
    # primeira infância — Heckman aplicado, LEPES/USP). Calibração inicial.
    k_saude_arr = k_hum_arr * 0.95

    # PIB estadual inicial (R$ bilhões): soma do pib_total zonal / 1e9
    pib_zonal = df_uf["pib_total"].to_numpy().astype(float)
    pib_estadual_inicial = float(pib_zonal.sum() / 1e9)

    # K_priv inicial (R$ bilhões): proxy via razão capital/produto K/Y ≈ 3.0
    # (literatura brasileira: Ferreira & Veloso 2013, IPEA estimativas).
    k_priv_inicial = pib_estadual_inicial * 3.0

    state = WorldState(
        t=0,
        R=R0_Gbbl,
        E_amb=float(df_uf["e_amb"].fillna(0.05).iloc[0]) if "e_amb" in df_uf else 0.05,
        P_efetiva=0.0,
        preco=preco0_usd,
        roy_periodo=0.0,
        gini=0.0,
        W=0.0,
        fundo_soberano=0.0,
        K_priv=k_priv_inicial,
        pib_estadual=pib_estadual_inicial,
        outras_receitas=5.0,  # piso inicial R$ bi/2 anos (IPVA + ITCMD + taxas + SUS)
        K_pub=df_uf["k_pub"].to_numpy().astype(float),
        K_hum=k_hum_arr,
        K_saude=k_saude_arr,
        N=n_pop_milhares,
        # C_inst inicial pode ser multiplicado pelo cenário (transformador
        # MA-Próspero usa ×5, ~0.30; baseline = 1.0, ~0.05-0.07).
        C_inst=np.clip(
            df_uf["c_inst"].to_numpy().astype(float) * c_inst_inicial_multiplier,
            0.0, 1.0,
        ),
        # Calibração inicial S_pressao_terr: valor não-zero refletindo conflitos
        # CPT pré-MEB acumulados no MA (2020-2023, ~200/ano). Valor 0.4 zonal
        # médio reproduz S₀ que vê salto +75% até 2024 quando MEB intensifica.
        S_pressao_terr=np.full(N_ZONES, 0.40),
    )
    # E_amb estadual: média ponderada por população das zonas
    state.E_amb = float(state.avg_pop(df_uf["e_amb"].fillna(0.05).to_numpy()))
    return state
