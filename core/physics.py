"""Equações de transição puras (spec §2.2 e §2.3).

Funções aqui não conhecem ``WorldState`` — recebem arrays/floats e devolvem
arrays/floats. Isso facilita teste unitário, paralelização e diferenciação
futura. ``World.step()`` orquestra estas funções com o estado e ações.

Convenções:
- ``dt`` em anos. Step da simulação = 2 anos (spec §2.1), então dt=2.
- Vetores zonais sempre shape (4,) na ordem ZONES de ``state.py``.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Parâmetros padrão (spec §2.2)
# ---------------------------------------------------------------------------
ETA_MAX = 0.8
"""Eficiência máxima de conversão de investimento em capital (spec §2.2.2)."""

DELTA_PUB = 0.04
"""Depreciação anual do capital público (spec §2.2.2)."""

DELTA_HUM = 0.02
"""Depreciação anual do capital humano (spec §2.2.3)."""

GAMMA_APREND = 0.02
"""Taxa de aprendizado institucional anual (spec §2.2.5)."""

GAMMA_CAPTURA = 0.02
"""Coeficiente de captura política por unidade de rent (spec §2.2.5).

Calibrado para que rent_per_cap normalizado ~0.5 reduza C_inst em ~10pp por
step de 2 anos — alinhando com observações de Mossoró/Macaé pós-boom.
"""

ALPHA_PROD = 0.03
"""Coeficiente de impacto ambiental EVITÁVEL por Mbbl/d (spec §2.2.6 v3 D').
Calibrado para que steady-state E_amb ∈ [0.1, 0.9] dependendo das ações
dos agentes (range fenomenologicamente realista para Margem Equatorial)."""

ALPHA_BASELINE_AMB = 0.02
"""Impacto ambiental IRREDUTÍVEL por Mbbl/d (spec §2.2.6 v3 D').
Operações offshore sempre deixam passivo: descartes legais (water-cut,
fluidos de perfuração), descarbonatação ácida — não chegam a zero mesmo
com máxima fiscalização. Calibração: Petrobras Relatório Sustentabilidade
2023 — passivos ambientais ~3% da receita anual recorrente."""

THETA_REMED = 0.20
"""Taxa de remediação efetiva por unidade de investimento ambiental.
Eficácia 30-60% reportada por empresas brasileiras de remediação
(Conestoga-Rovers Brasil; BIO-RICS)."""

DECAIMENTO_NATURAL_AMB = 0.02
"""Atenuação ecossistêmica natural anual (spec §2.2.6 v3 D').
~2%/ano de "auto-remediação" (correntes oceânicas, biodegradação,
diluição). Calibrado pelos estudos de manguezais brasileiros — meia-vida
~5-10 anos para poluentes orgânicos sob condições naturais."""

G_NATURAL = 0.008
"""Crescimento populacional natural anual (spec §2.2.4 para MA)."""

W_WEIGHTS = (0.25, 0.30, 0.20, 0.15, 0.10)
"""Pesos de W: (K_pub, K_hum, log_renda, E_amb, Gini) — spec §2.3.2."""


# ---------------------------------------------------------------------------
# §2.2.1 — Reservas e produção (Hubbert modulado)
# ---------------------------------------------------------------------------
def hubbert_production(
    t_step: int,
    P_max_mbbld: float,  # noqa: N803 -- nome matemático da spec
    t_peak_step: int,
    a_assimetria: float,
) -> float:
    """Produção potencial seguindo Hubbert modificado (spec §2.2.1).

    P(t) = P_max * (4 * e^(-a*(t-t_peak))) / (1 + e^(-a*(t-t_peak)))^2

    Returns:
        Produção em Mbbl/d (milhões de barris/dia).
    """
    delta = float(t_step - t_peak_step)
    e_neg = np.exp(-a_assimetria * delta)
    return float(P_max_mbbld * (4.0 * e_neg) / (1.0 + e_neg) ** 2)


def producao_efetiva(
    p_potencial_mbbld: float,
    alpha_investimento: float,
    R_atual_Gbbl: float,  # noqa: N803
) -> float:
    """Produção efetiva = potencial × ação da Operadora, limitada por reservas.

    Args:
        p_potencial_mbbld: produção Hubbert potencial.
        alpha_investimento: ação da Operadora ∈ [0.5, 1.2].
        R_atual_Gbbl: reservas restantes; se ≤ 0, P=0 (spec §2.2.1).
    """
    if R_atual_Gbbl <= 0:
        return 0.0
    return p_potencial_mbbld * float(np.clip(alpha_investimento, 0.5, 1.2))


def update_reservas(
    R_atual_Gbbl: float,  # noqa: N803
    p_efetiva_mbbld: float,
    dt_anos: float,
) -> float:
    """R(t+1) = R(t) - P_efetiva * dt, em Gbbl.

    1 Mbbl/d * 365 dias = 365 Mbbl/ano = 0.365 Gbbl/ano.
    """
    consumo_gbbl = p_efetiva_mbbld * 365 * dt_anos / 1000.0
    return max(0.0, R_atual_Gbbl - consumo_gbbl)


# ---------------------------------------------------------------------------
# §2.2.2 — Capital público (zonal)
# ---------------------------------------------------------------------------
def eta_efetivo(c_inst_zone: np.ndarray, eta_max: float = ETA_MAX) -> np.ndarray:
    """Eficiência de conversão mediada por capacidade institucional."""
    return eta_max * c_inst_zone


def update_K_pub(  # noqa: N802
    k_pub_zone: np.ndarray,
    i_infra_zone: np.ndarray,
    c_inst_zone: np.ndarray,
    dt_anos: float = 2.0,
    delta_pub: float = DELTA_PUB,
) -> np.ndarray:
    """K_pub[z](t+1) = K_pub[z] + η[z] * I_infra[z] * dt - δ_pub * K_pub[z] * dt."""
    eta = eta_efetivo(c_inst_zone)
    novo = k_pub_zone + (eta * i_infra_zone - delta_pub * k_pub_zone) * dt_anos
    return np.clip(novo, 0.0, 1.0)


# ---------------------------------------------------------------------------
# §2.2.3 — Capital humano (zonal, com defasagem τ)
# ---------------------------------------------------------------------------
def update_K_hum(  # noqa: N802
    k_hum_zone: np.ndarray,
    i_educ_lag_zone: np.ndarray,  # investimento defasado τ steps atrás
    c_inst_zone: np.ndarray,
    epsilon_emigra_zone: np.ndarray,
    dt_anos: float = 2.0,
    delta_hum: float = DELTA_HUM,
) -> np.ndarray:
    """K_hum[z](t+1) inclui depreciação natural + emigração."""
    eta = eta_efetivo(c_inst_zone)
    novo = (
        k_hum_zone
        + (eta * i_educ_lag_zone - (delta_hum + epsilon_emigra_zone) * k_hum_zone) * dt_anos
    )
    return np.clip(novo, 0.0, 1.0)


def update_K_saude(  # noqa: N802
    k_saude_zone: np.ndarray,
    i_saude_lag_zone: np.ndarray,  # investimento defasado τ=2 steps (4 anos)
    c_inst_zone: np.ndarray,
    epsilon_emigra_zone: np.ndarray,
    dt_anos: float = 2.0,
    delta_saude: float = 0.05,  # Grossman 1972; calibração via DATASUS
    eta_saude: float = 0.50,
) -> np.ndarray:
    """K_saude[z](t+1) — modelo Grossman 1972 + WHO Building Blocks.

    Spec §2.3.6: capital saúde populacional como stock próprio. Lag τ=2 steps
    (4 anos) — menor que K_hum (8 anos) porque atenção primária e cobertura
    vacinal mostram efeito em janelas de 24-48 meses (literatura SUS).

    Eficiência mediada por C_inst (governance, WHO building block #1).
    """
    eta = eta_saude * eta_efetivo(c_inst_zone) / ETA_MAX  # normaliza pelo max
    novo = (
        k_saude_zone
        + (eta * i_saude_lag_zone - (delta_saude + epsilon_emigra_zone) * k_saude_zone) * dt_anos
    )
    return np.clip(novo, 0.0, 1.0)


def aplica_mobilizacao(
    mobilizacao: float,
    pref_amb: float,
    phi_fisc_reg: float,
    rent_per_capita_zone: np.ndarray,
    n_zone: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mobilização da comunidade afeta física via 3 canais (spec §3.5 v2).

    Calibração brasileira:
    - +25% em phi_fisc efetivo quando mob × pref_amb = 1
      (Zhouri & Laschefski UFMG 2010, casos Belo Monte/Carajás-EFC).
    - −30% em captura por rent quando mob = 1
      (Avritzer 2009 sobre conselhos participativos).

    Returns:
        (phi_eff, rent_eff): fiscalização efetiva e rent atenuado por zona.
    """
    m = float(np.clip(mobilizacao, 0.0, 1.0))
    p = float(np.clip(pref_amb, 0.0, 1.0))

    # (a) Fiscalização efetiva: regulador responde mais sob holofotes ambientais
    phi_eff = float(np.clip(phi_fisc_reg + 0.25 * m * p, 0.0, 1.0))

    # (b) Rent atenuado: pressão organizada ↓ captura, proporcional à zona mobilizada
    share_zone = n_zone / max(n_zone.sum(), 1e-9)
    rent_eff = rent_per_capita_zone * (1.0 - 0.30 * m * share_zone)

    return phi_eff, rent_eff


def custo_mobilizacao(
    mobilizacao: float,
    mob_hist: float,
    repressao_idx: float = 0.10,
    lambda_mob: float = 0.08,
) -> float:
    """Custo convexo + fadiga + risco de repressão (spec §3.5 v2).

    Forma funcional:
    - Convexo m² (Olson 1965 + Tarrow): cada ponto de mob extra custa mais.
    - Fadiga: mobilização sustentada 0.5·m·mob_hist (Alonso CEBRAP 2009).
    - Risco repressão: repressao_idx·m·(0.5+mob_hist) (CPT 2023, MA pós-Carajás).

    Returns:
        Custo escalar para subtrair de R_com.
    """
    m = float(np.clip(mobilizacao, 0.0, 1.0))
    h = float(np.clip(mob_hist, 0.0, 1.0))
    convexo = m**2
    fadiga = 0.5 * m * h
    risco = repressao_idx * m * (0.5 + h)
    return float(lambda_mob * (convexo + fadiga + risco))


def update_K_priv(  # noqa: N802
    k_priv: float,
    receita_op_bi: float,  # R$ bi do período
    share_invest_local: float = 0.30,  # SHARE_RECEITA_OP_INVEST_LOCAL
    delta_kpriv: float = 0.08,  # DELTA_KPRIV_ANUAL
    dt_anos: float = 2.0,
) -> float:
    """K_priv(t+1) — capital privado estadual em R$ bilhões.

    Spec §2.3.4: alimentado por fração da receita líquida da operadora
    reinvestida localmente (Petrobras Plano Estratégico declara ~30% de
    OPEX em fornecedores locais — Conteúdo Local ANP). Depreciação anual
    8% (Ferreira & Veloso 2013, IPEA).
    """
    investimento = share_invest_local * receita_op_bi
    novo = k_priv * (1.0 - delta_kpriv) ** dt_anos + investimento * dt_anos
    return float(max(novo, 0.0))


# ---------------------------------------------------------------------------
# §2.3.4 — PIB estadual endógeno (Cobb-Douglas)
# ---------------------------------------------------------------------------
def compute_pib_estadual(
    k_priv: float,  # R$ bi
    avg_k_pub: float,  # [0,1]
    avg_k_hum: float,  # [0,1]
    avg_k_saude: float,  # [0,1]
    n_total_milhares: float,
    a_tfp: float = 0.25,
    alpha: float = 0.30,
    beta: float = 0.12,
    gamma: float = 0.30,
    delta: float = 0.10,
    theta: float = 0.18,
) -> float:
    """PIB estadual em R$ bilhões via função Cobb-Douglas com 5 fatores.

    Spec §2.3.4: PIB = A · K_priv^α · K_pub^β · K_hum^γ · K_saude^δ · N^θ
    Σ exponentes = 1 (rendimentos constantes de escala).

    K_priv em R$ bi; demais em índices [0,1] ou milhares de habitantes.
    A_TFP calibrado de modo a reproduzir PIB MA observado (~R$ 109 bi 2022)
    nas condições iniciais.
    """
    # Floor pequeno para evitar 0^β (=0) quando algum capital é zero.
    k_priv_safe = max(k_priv, 1e-6)
    kp = max(avg_k_pub, 1e-6)
    kh = max(avg_k_hum, 1e-6)
    ks = max(avg_k_saude, 1e-6)
    n = max(n_total_milhares, 1e-6)
    pib = a_tfp * (k_priv_safe**alpha) * (kp**beta) * (kh**gamma) * (ks**delta) * (n**theta)
    return float(pib)


# ---------------------------------------------------------------------------
# §2.3.5 — Receitas estaduais (RCL decomposta)
# ---------------------------------------------------------------------------
def step_icms(
    pib_estadual_bi: float,
    aliquota_efetiva: float = 0.10,
    epsilon: float = 1.10,
    pib_referencia_bi: float = 109.0,  # MA 2022 IBGE
) -> float:
    """ICMS endógeno ao PIB estadual (spec §2.3.5).

    Forma: ICMS = (alíquota × PIB_ref) · (PIB / PIB_ref)^ε
    Elasticidade ε > 1 (PPP/IPEA 2024 — caso ES; padrão se replica em UFs).
    Resultado em R$ bi/ano (multiplicar por dt_anos no caller para período).
    """
    base_anual = aliquota_efetiva * pib_referencia_bi
    fator = (pib_estadual_bi / max(pib_referencia_bi, 1e-9)) ** epsilon
    return float(base_anual * fator)


def step_fpe(
    pool_ir_ipi_bi: float,
    cota_uf: float = 0.072,  # MA — LC 143/2013
    share_estadual_pool: float = 0.215,  # 21.5% IR+IPI vai para FPE
) -> float:
    """FPE da UF em R$ bi/ano. Cota fixa por LC 143/2013."""
    return float(cota_uf * share_estadual_pool * pool_ir_ipi_bi)


def step_pool_ir_ipi(
    pool_anterior_bi: float,
    crescimento_anual: float = 0.025,
    dt_anos: float = 2.0,
) -> float:
    """Pool nacional IR+IPI cresce com PIB nacional (cenário macro)."""
    return float(pool_anterior_bi * (1.0 + crescimento_anual) ** dt_anos)


def step_outras_receitas(
    outras_anterior_bi: float,
    drift_anual: float = 0.025,
    dt_anos: float = 2.0,
) -> float:
    """IPVA + ITCMD + taxas + SUS — drift exógeno alinhado com PIB nacional."""
    return float(outras_anterior_bi * (1.0 + drift_anual) ** dt_anos)


def shift_lag_buffer(buffer: np.ndarray, novo: np.ndarray) -> np.ndarray:
    """Avança o buffer de defasagem (τ=4 colunas) introduzindo o novo investimento.

    Args:
        buffer: shape (n_zones, tau)
        novo: shape (n_zones,) - investimento corrente

    Returns:
        novo buffer shape (n_zones, tau): coluna 0 = mais antigo (sai do retorno),
        última coluna recebe ``novo``. O retorno do ``novo[]`` passado como
        ``i_educ_lag_zone`` é a coluna 0 (índice 0) — que é o investimento
        feito τ steps atrás.
    """
    rolled = np.roll(buffer, shift=-1, axis=1)
    rolled[:, -1] = novo
    return rolled


# ---------------------------------------------------------------------------
# §2.2.4 — População (zonal, gravitacional)
# ---------------------------------------------------------------------------
def migracao_externa(
    w_local: np.ndarray,
    w_nacional: float,
    M_max: float = 50.0,  # noqa: N803
    beta: float = 2.0,
) -> np.ndarray:
    """Fluxo migratório externo (in/out da UF) por zona, em milhares/período."""
    diff = w_local / max(w_nacional, 1e-9) - 1.0
    return M_max * (1.0 / (1.0 + np.exp(-beta * diff)) - 0.5) * 2.0  # range ~[-M_max, M_max]


def update_N(  # noqa: N802
    n_zone: np.ndarray,
    w_local_zone: np.ndarray,
    w_nacional: float,
    dt_anos: float = 2.0,
    g_natural: float = G_NATURAL,
) -> np.ndarray:
    """População por zona com crescimento natural + migração externa.

    Migração intra-UF (entre zonas) é modelada à parte; aqui só o saldo
    com o exterior da UF.
    """
    natural = n_zone * (1.0 + g_natural * dt_anos)
    m_ext = migracao_externa(w_local_zone, w_nacional)
    return np.maximum(0.0, natural + m_ext * dt_anos)


# ---------------------------------------------------------------------------
# §2.2.5 — Capacidade institucional (zonal, com captura)
# ---------------------------------------------------------------------------
def update_C_inst(  # noqa: N802
    c_inst_zone: np.ndarray,
    i_gov_zone: np.ndarray,
    rent_per_capita_zone: np.ndarray,
    dt_anos: float = 2.0,
    gamma_aprend: float = GAMMA_APREND,
    gamma_captura: float = GAMMA_CAPTURA,
) -> np.ndarray:
    """C_inst[z](t+1) = C_inst[z] + γ_aprend*I_gov[z]*dt - γ_captura*Rent[z]*dt.

    Loop central de resource-curse: rent alto reduz C_inst, que reduz η,
    que reduz eficiência de TODOS os investimentos (spec §2.2.5).
    """
    delta = gamma_aprend * i_gov_zone - gamma_captura * rent_per_capita_zone
    novo = c_inst_zone + delta * dt_anos
    return np.clip(novo, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Pressão Territorial (H-TERR-2 — calibração CPT/INCRA/CIMI 2024)
# ---------------------------------------------------------------------------
# Substitui placeholder H-TERR-1 (`territorio_proxy = 1 - 0.5·E_amb`).
# Forma logística com histerese, fundamentada em:
# - CPT (2025): Maranhão saltou de 200/ano (2020-23) → 363 conflitos em 2024 (+75%),
#   1º nacional. Andersen-Nordvik-Tesei (2022) sobre choques de preço offshore.
# - Watts (2004) Niger Delta: assimetria renda-central / externalidades-locais.
# - Escobar (2008), Almeida (NAEA/UFPA 2008): path-dependence de conflitos.
# - INCRA-MA: 3 títulos de 424 processos quilombolas → C_inst_baseline ≈ 0.007.
# Calibração-sanidade MA 2024: σ(1.5 - 1.0·0.6 - 1.5·0.5 - 2.0·(1-0.05) - 0.8·0.4) ≈ 0.10
# (vs. 0.70 do placeholder antigo — superdimensionava estabilidade em 7×).

DELTA_PRESSAO_DECAY = 0.03
"""Decaimento mensal de S_pressao (path-dependence: meia-vida ~23 períodos)."""

LAMBDA_PRESSAO_ACUM = 0.4
"""Coeficiente de acumulação por novo choque (ΔE_amb ou rent)."""

OMEGA_PRESSAO_RENT = 0.6
"""Peso da rent extrativa não-capturada na geração de pressão (Watts 2004)."""

TERR_INTERCEPT = 1.5
"""Intercepto a do logit de territorio_proxy."""

TERR_ALPHA_EAMB = 1.0
"""Coeficiente α: peso da degradação ambiental atual."""

TERR_BETA_RENT = 1.5
"""Coeficiente β: peso da rent extrativa não-capturada (Niger Delta-like)."""

TERR_GAMMA_INST = 2.0
"""Coeficiente γ: peso do déficit institucional (cliff INCRA-MA)."""

TERR_ETA_HIST = 0.8
"""Coeficiente η: peso do estoque histórico (histerese Escobar/Almeida)."""


def update_S_pressao(  # noqa: N802 -- nome matemático (S = stock)
    s_pressao_zone: np.ndarray,
    delta_e_amb: float,
    rent_per_capita_zone: np.ndarray,
    dt_anos: float = 2.0,
    delta_decay: float = DELTA_PRESSAO_DECAY,
    lambda_acum: float = LAMBDA_PRESSAO_ACUM,
    omega_rent: float = OMEGA_PRESSAO_RENT,
) -> np.ndarray:
    """S_z(t+1) = (1 − δ)·S_z + λ·max(0, ΔE_amb + ω·rent_z) (zonal, histerese).

    Captura path-dependence dos conflitos territoriais — calibração CPT 2020-2024:
    o salto MA 2023→2024 (+75%) reflete acumulação de 3 anos de choques.
    """
    decay_factor = (1.0 - delta_decay) ** dt_anos
    choque = np.maximum(0.0, delta_e_amb + omega_rent * rent_per_capita_zone)
    novo = decay_factor * s_pressao_zone + lambda_acum * choque
    return np.clip(novo, 0.0, 1.0)


def compute_territorio_proxy_zonal(
    e_amb: float,
    rent_per_capita_zone: np.ndarray,
    c_inst_zone: np.ndarray,
    s_pressao_zone: np.ndarray,
    intercept: float = TERR_INTERCEPT,
    alpha_eamb: float = TERR_ALPHA_EAMB,
    beta_rent: float = TERR_BETA_RENT,
    gamma_inst: float = TERR_GAMMA_INST,
    eta_hist: float = TERR_ETA_HIST,
) -> np.ndarray:
    """territorio_proxy_z = σ(a − α·E_amb − β·R_z − γ·(1−C_inst_z) − η·S_z).

    Forma logística com histerese (H-TERR-2). Retorna vetor zonal ∈ [0, 1].
    Valor alto = território estável. Valor baixo = invasão/conflito.
    """
    logit = (
        intercept
        - alpha_eamb * float(e_amb)
        - beta_rent * np.asarray(rent_per_capita_zone, dtype=float)
        - gamma_inst * (1.0 - np.asarray(c_inst_zone, dtype=float))
        - eta_hist * np.asarray(s_pressao_zone, dtype=float)
    )
    return 1.0 / (1.0 + np.exp(-logit))


# ---------------------------------------------------------------------------
# §2.2.6 — Passivo ambiental (escalar UF)
# ---------------------------------------------------------------------------
def prob_acidente(
    p_efetiva_mbbld: float,
    phi_fisc: float,
    lambda_base: float,
    alpha_seg: float = 0.0,
    beta_seg: float = 0.6,
    rigor_seg_anp: float = 0.0,
    beta_anp: float = 0.5,
) -> float:
    """λ_acid = λ_base · P · (1−φ_fisc) · (1−β_seg·α_seg) · exp(−β_anp·rigor_seg).

    Spec §2.2.6 + §3.3 + Res. ANP 882/2022 (PSO):
    - α_seg (operadora): redução até β_seg=0.6 quando α_seg=1.
    - rigor_seg_anp (ANP): exigência de Programa de Segurança Operacional.
      Reduz prob. multiplicativamente via exp(−β_anp·rigor) — calibração
      β_anp=0.5 alinhada a redução observada de ~40% em incidentes pós-PSO
      em campos sob fiscalização rigorosa (TCU Acórdão 2.936/2021).
    """
    fisc_factor = 1.0 - float(np.clip(phi_fisc, 0.0, 1.0))
    seg_factor = 1.0 - beta_seg * float(np.clip(alpha_seg, 0.0, 1.0))
    anp_factor = float(np.exp(-beta_anp * float(np.clip(rigor_seg_anp, 0.0, 1.0))))
    return lambda_base * p_efetiva_mbbld * fisc_factor * seg_factor * anp_factor


def update_E_amb(  # noqa: N802
    e_amb: float,
    p_efetiva_mbbld: float,
    phi_fisc: float,
    i_amb: float,
    acidente_severidade: float,
    dt_anos: float = 2.0,
    alpha_prod: float = ALPHA_PROD,
    alpha_baseline: float = ALPHA_BASELINE_AMB,
    theta_remed: float = THETA_REMED,
    decaimento_natural: float = DECAIMENTO_NATURAL_AMB,
) -> float:
    """Evolui o passivo ambiental (spec §2.2.6 v3 — calibração D').

    Forma:
        ΔE_amb/dt = α_baseline·P                              (irreduzível)
                  + α_prod·P·(1 − φ_fisc)                      (evitável)
                  − θ_remed·I_amb·E_amb                        (remediação ∝ E_amb)
                  − γ_natural·E_amb                            (decaimento)
        E_amb(t+1) = clip(E_amb + ΔE_amb·dt + severidade, 0, 1)

    **Mudança estrutural v3 (D'):** remediação agora multiplica por `E_amb`.
    Princípio fenomenológico: não faz sentido "remediar onde não há passivo".
    Resolve a saturação artificial em 0 da v2 sem mudar incentivos.

    Steady-state com P=1, α_seg=1, φ_fisc=1:
        0 = 0.02 − 0.04·E − 0.02·E ⇒ E* ≈ 0.33 (realista)
    Com φ_fisc=0 (negligência): E* ≈ 0.83 (degradação alta)
    """
    p = float(p_efetiva_mbbld)
    phi = float(np.clip(phi_fisc, 0.0, 1.0))
    impacto_base = alpha_baseline * p
    impacto_evit = alpha_prod * p * (1.0 - phi)
    remediacao = theta_remed * i_amb * e_amb  # v3 D': proporcional a E_amb
    decaimento = decaimento_natural * e_amb
    delta = impacto_base + impacto_evit - remediacao - decaimento
    novo = e_amb + delta * dt_anos + acidente_severidade
    return float(np.clip(novo, 0.0, 1.0))


# ---------------------------------------------------------------------------
# §2.3.1 — Royalties
# ---------------------------------------------------------------------------
def royalties_periodo(
    p_efetiva_mbbld: float,
    preco_usd: float,
    tau_royalty: float,
    cambio_brl_usd: float = 5.0,
    dt_anos: float = 2.0,
) -> float:
    """Royalties em R$ no período: P * Preço * τ_royalty * dias * cambio."""
    receita_usd = p_efetiva_mbbld * 1e6 * preco_usd * 365 * dt_anos
    return receita_usd * tau_royalty * cambio_brl_usd


# ---------------------------------------------------------------------------
# §2.3.2 — Indicadores agregados (W, Gini)
# ---------------------------------------------------------------------------
def gini_zonal(renda_pc_zone: np.ndarray, n_zone: np.ndarray) -> float:
    """Índice de Gini ponderado por população (spec §2.6).

    0 = perfeita igualdade, 1 = concentração máxima.
    """
    if n_zone.sum() <= 0:
        return 0.0
    order = np.argsort(renda_pc_zone)
    v = renda_pc_zone[order]
    w = n_zone[order]
    total_w = w.sum()
    mean = float(np.dot(v, w) / total_w)
    if mean <= 0:
        return 0.0
    cumw = np.cumsum(w)
    return float(np.sum((2 * cumw - w) * v * w) / (total_w**2 * mean) - 1)


def bem_estar(
    k_pub_zone: np.ndarray,
    k_hum_zone: np.ndarray,
    n_zone: np.ndarray,
    renda_pc_uf: float,
    e_amb: float,
    gini: float,
    k_saude_zone: np.ndarray | None = None,
    weights: tuple[float, ...] | None = None,
) -> float:
    """W(t) agregado spec §2.3.2 v2 (versão zonal §2.6 com K_saude).

    Forma v2:
        W = w1·avg_pop(K_pub) + w2·avg_pop(K_hum) + w3·avg_pop(K_saude)
            + w4·log(renda_pc) − w5·E_amb − w6·Gini

    Pesos default v2 (Heckman/IDH-M decomposition): (0.20, 0.25, 0.20, 0.15,
    0.10, 0.10). Para retrocompatibilidade com chamadas v1 (5 pesos), se
    k_saude_zone não for passado, omite o termo de saúde.
    """
    pop_total = max(n_zone.sum(), 1e-9)
    avg_kpub = float(np.dot(k_pub_zone, n_zone) / pop_total)
    avg_khum = float(np.dot(k_hum_zone, n_zone) / pop_total)
    log_renda = float(np.log(max(renda_pc_uf, 1.0)))

    if k_saude_zone is not None:
        # v2: 6 termos
        if weights is None:
            weights = (0.20, 0.25, 0.20, 0.15, 0.10, 0.10)
        w1, w2, w3, w4, w5, w6 = weights
        avg_ksaude = float(np.dot(k_saude_zone, n_zone) / pop_total)
        return float(
            w1 * avg_kpub
            + w2 * avg_khum
            + w3 * avg_ksaude
            + w4 * log_renda
            - w5 * e_amb
            - w6 * gini
        )
    # v1 retrocompat: 5 pesos
    if weights is None:
        weights = W_WEIGHTS
    w1, w2, w3, w4, w5 = weights
    return float(w1 * avg_kpub + w2 * avg_khum + w3 * log_renda - w4 * e_amb - w5 * gini)


# ---------------------------------------------------------------------------
# §2.4 — Preço estocástico
# ---------------------------------------------------------------------------
def step_preco(
    preco_atual: float,
    sigma_anual: float,
    mu_drift_anual: float,
    dt_anos: float,
    rng: np.random.Generator,
) -> float:
    """log(P_{t+1}) = log(P_t) + μ*dt + σ*sqrt(dt)*ε."""
    eps = float(rng.normal())
    log_novo = (
        np.log(max(preco_atual, 1.0))
        + mu_drift_anual * dt_anos
        + sigma_anual * np.sqrt(dt_anos) * eps
    )
    return float(np.exp(log_novo))
