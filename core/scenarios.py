"""Cenários exógenos da simulação (spec §2.4).

Os parâmetros são calibrados a partir dos dados ingeridos:
- Brent inicial e volatilidade <- EIA Open Data
- Drift de preço de longo prazo <- EIA + EPE PDE 2034 (transição energética)
- VA Extrativa BR <- EPE PDE 2034 Caderno de Premissas Econômicas
- URR (reservas) <- ANP / EPE Zoneamento (cenários P5/P50/P95)
- P_max e t_peak (Hubbert) <- ANP histórico de produção brasileira
- Macroeconomia estadual (RCL, PIB) <- STN/RREO MA, IPEA TD 2782, PPP/IPEA 2024
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constantes ESTRUTURAIS (não variam entre cenários macro — são institucionais
# ou empíricas estáveis no Brasil pós-pré-sal). Spec §2.3.4-§2.3.6.
# ---------------------------------------------------------------------------

# Cota FPE do MA (LC 143/2013, Cartilha Tesouro). Estado pobre, alta dependência.
COTA_FPE_MA: float = 0.072

# Alíquota efetiva ICMS sobre PIB estadual no MA (STN 2023: ICMS R$10.9 bi /
# PIB MA R$109 bi ≈ 0.10). Aproximação MA-específica.
ALIQUOTA_ICMS_EFETIVA: float = 0.10

# Elasticidade ICMS-PIB no longo prazo (PPP/IPEA 2024 — caso ES, replicável MA).
EPSILON_ICMS_PIB: float = 1.10

# Função de produção PIB estadual (Cobb-Douglas, rendimentos constantes):
# PIB = A · K_priv^α · K_pub^β · K_hum^γ · K_saude^δ · N^θ ,  Σ = 1
# Calibrado de Ferreira & Veloso (IPEA), Heckman aplicado ao Brasil.
ALPHA_PIB_KPRIV: float = 0.30
BETA_PIB_KPUB: float = 0.12
GAMMA_PIB_KHUM: float = 0.30
DELTA_PIB_KSAUDE: float = 0.10
THETA_PIB_N: float = 0.18  # Σ = 1.0
A_TFP: float = 6.0
"""Produtividade total dos fatores. Calibrada para reproduzir PIB MA observado
(~R$ 125 bi 2022, IBGE) nas condições iniciais (K_priv≈375bi, K_pub≈0.76,
K_hum≈0.556, K_saude≈0.528, N≈7153 milhares). A_TFP × prod(stocks^expoentes)
= PIB_alvo. Validar antes de outras UFs (AP/PA/RN têm A_TFP próprio)."""

# Capital privado: depreciação anual (Ferreira & Veloso 2013) e fração da
# receita líquida da operadora reinvestida como capital privado local.
DELTA_KPRIV_ANUAL: float = 0.08
SHARE_RECEITA_OP_INVEST_LOCAL: float = 0.30

# Capital saúde (Grossman 1972, calibração via DATASUS): depreciação menor que
# K_pub porque "estoque saúde populacional" é mais lento.
DELTA_KSAUDE_ANUAL: float = 0.05
ETA_KSAUDE: float = 0.50  # eficiência conversão investimento → estoque

# Pesos do bem-estar W (spec §2.3.2 v2.0). IDH-M brasileiro decompõe ~1/3
# educação, ~1/3 longevidade, ~1/3 renda; adaptado.
W_WEIGHTS_V2: tuple[float, float, float, float, float, float] = (
    0.20,  # K_pub
    0.25,  # K_hum
    0.20,  # K_saude (novo)
    0.15,  # log(renda_pc)
    0.10,  # E_amb (penalidade)
    0.10,  # Gini (penalidade)
)

# FUNDEB: complementação federal líquida que MA recebe (EC 108/2020, regra
# atualizada para 23% até 2026). Valor anual em R$ bilhões — relativamente
# estável, semi-determinístico.
FUNDEB_LIQUIDO_MA_BI: float = 6.0  # ~R$ 6 bi/ano (STN 2023)

# Pool nacional IR+IPI inicial (R$ bi, base 2024 SEFAZ/STN).
POOL_IR_IPI_INICIAL_BI: float = 580.0
"""Pool nacional IR + IPI base (R$ bi 2024). Calibrado para reproduzir
FPE-MA empírico ~R$ 8-9 bi/ano (literatura: STN/RREO 2023, Pellegrini/FGV).
Receita Federal IR 2023 ~R$ 700 bi + IPI ~R$ 70 bi líquidos de desonerações
e retenções federais ≈ R$ 580 bi efetivamente compartilháveis. FPE-MA
calculado: 580 × 0.215 × 0.072 ≈ 8.97 bi/ano."""

# Lei 12.858/2013 — vinculação obrigatória dos royalties novos.
SHARE_ROY_EDUC_LEI_12858: float = 0.75
SHARE_ROY_SAUDE_LEI_12858: float = 0.25


@dataclass(frozen=True)
class Scenario:
    """Pacote de parâmetros exógenos para uma rodada de simulação."""

    nome: str

    # Reservas e produção (spec §2.2.1)
    URR_Gbbl: float
    """Ultimate Recoverable Reserves em giga-barris."""

    P_max_mbbld: float
    """Produção de pico em milhões de barris/dia (Hubbert P_max)."""

    t_peak_step: int
    """Step do pico (0..15) para a curva de Hubbert."""

    a_assimetria: float
    """Fator de assimetria da curva (>1 = ramp-up rápido)."""

    # Preço (spec §2.4)
    preco0_usd: float
    """Preço Brent inicial USD/bbl."""

    sigma_preco: float
    """Volatilidade anual log-retorno (~0.15-0.30 spec; 0.48 EIA recente)."""

    mu_drift: float
    """Drift anual log de preço (negativo em transição energética acelerada)."""

    # Macro (alinha com EPE PDE 2034)
    crescimento_w_nacional: float
    """Crescimento real anual da renda per capita nacional (1-2% spec)."""

    # Royalties (Lei 12.351/2010)
    tau_royalty: float
    """Alíquota efetiva royalties + PE (0.15-0.20)."""

    # Acidentes (spec §2.2.6)
    lambda_acid_base: float
    """Probabilidade base de acidente por step."""

    # Crescimento real anual do pool IR+IPI nacional (afeta FPE de todas UFs).
    # Correlacionado com PIB nacional do cenário macro.
    crescimento_ir_ipi_anual: float = 0.025

    # ----- Contrafactuais (regime legal) -----
    vinculacao_lei_12858: bool = True
    """Quando False, royalties não são vinculados (75/25 educ/saúde) — viram
    RCL livre. Modela revogação simulada da Lei 12.858/2013."""

    vinculacao_outras_receitas: float = 0.0
    """Fração das 'outras receitas' vinculadas adicionalmente a infra+saúde
    (cenário transformador MA-Próspero, hipotético reforço estatutário)."""

    # ----- Choque exógeno de preço -----
    price_shock_step: int | None = None
    """Step em que aplica choque de preço; None = sem choque."""

    price_shock_factor: float = 1.0
    """Multiplicador aplicado ao preço quando t == price_shock_step
    (ex: 0.6 = queda de 40%)."""

    # ----- Cenário transformador (MA-Próspero) -----
    c_inst_inicial_multiplier: float = 1.0
    """Multiplicador aplicado a C_inst inicial (cenário transformador
    'aposta institucional' — MA começa com Estado mais maduro)."""

    delta_pressao_decay: float | None = None
    """Override de DELTA_PRESSAO_DECAY (None = usa padrão 0.03 de physics).
    Cenário transformador usa 0.06 (regularização fundiária ativa)."""

    lambda_pressao_acum: float | None = None
    """Override de LAMBDA_PRESSAO_ACUM (None = usa padrão 0.4).
    Cenário transformador usa 0.2 (proteção institucional reduzida acumulação)."""


# ---------------------------------------------------------------------------
# Cenários presets — calibrados com os dados reais ingeridos
# ---------------------------------------------------------------------------

SCENARIO_PESSIMISTA = Scenario(
    nome="pessimista",
    URR_Gbbl=3.0,  # P95 spec §2.4
    P_max_mbbld=0.5,
    t_peak_step=4,  # pico em 2036
    a_assimetria=1.4,
    preco0_usd=60.0,
    sigma_preco=0.30,
    mu_drift=-0.02,  # transição energética acelerada
    crescimento_w_nacional=0.016,  # EPE Cenário Inferior 2024-2034 (~1.6%)
    tau_royalty=0.15,
    lambda_acid_base=0.05,
    crescimento_ir_ipi_anual=0.016,  # acompanha PIB nacional pessimista
)

SCENARIO_REFERENCIA = Scenario(
    nome="referencia",
    URR_Gbbl=8.0,  # P50 spec §2.4
    P_max_mbbld=1.0,
    t_peak_step=6,  # pico em 2040
    a_assimetria=1.5,
    preco0_usd=80.0,  # Brent média 2020-2026 EIA
    sigma_preco=0.20,
    mu_drift=-0.005,
    crescimento_w_nacional=0.025,  # EPE Cenário Referência (~2.5% PIB var)
    tau_royalty=0.18,
    lambda_acid_base=0.03,
    crescimento_ir_ipi_anual=0.025,
)

SCENARIO_OTIMISTA = Scenario(
    nome="otimista",
    URR_Gbbl=15.0,  # P5 spec §2.4
    P_max_mbbld=2.0,
    t_peak_step=8,  # pico em 2044, longo platô
    a_assimetria=1.6,
    preco0_usd=100.0,
    sigma_preco=0.18,
    mu_drift=0.005,
    crescimento_w_nacional=0.037,  # EPE Cenário Superior 2034 (~3.7% a.a.)
    tau_royalty=0.20,
    lambda_acid_base=0.02,
    crescimento_ir_ipi_anual=0.037,
)


SCENARIO_SEM_LEI12858 = Scenario(
    nome="sem_lei12858",
    URR_Gbbl=8.0,
    P_max_mbbld=1.0,
    t_peak_step=6,
    a_assimetria=1.5,
    preco0_usd=80.0,
    sigma_preco=0.20,
    mu_drift=-0.005,
    crescimento_w_nacional=0.025,
    tau_royalty=0.18,
    lambda_acid_base=0.03,
    crescimento_ir_ipi_anual=0.025,
    # Contrafactual: revogação simulada da Lei 12.858/2013
    vinculacao_lei_12858=False,
)

SCENARIO_CHOQUE_BRENT = Scenario(
    nome="choque_brent",
    URR_Gbbl=8.0,
    P_max_mbbld=1.0,
    t_peak_step=6,
    a_assimetria=1.5,
    preco0_usd=80.0,
    sigma_preco=0.20,
    mu_drift=-0.005,
    crescimento_w_nacional=0.025,
    tau_royalty=0.18,
    lambda_acid_base=0.03,
    crescimento_ir_ipi_anual=0.025,
    # Choque exógeno: queda 40% no preço Brent no step 12 (meio do horizonte
    # 25 steps = 50 anos). Modela commodity bust à la 2014-2016.
    price_shock_step=12,
    price_shock_factor=0.6,
)

SCENARIO_MA_PROSPERO = Scenario(
    nome="ma_prospero",
    # Petróleo bem-aproveitado, não 'mais petróleo': mantém referência
    URR_Gbbl=8.0,
    P_max_mbbld=1.0,
    t_peak_step=6,
    a_assimetria=1.5,
    # Brent estável + crescimento moderado (Norway 2010s benchmark)
    preco0_usd=80.0,
    sigma_preco=0.15,  # menor volatilidade
    mu_drift=0.005,    # crescimento real moderado
    crescimento_w_nacional=0.030,  # PDE Cenário Superior
    crescimento_ir_ipi_anual=0.030,
    # Captura tributária máxima (teto Lei 9.478/97 art. 47)
    tau_royalty=0.25,
    # Operação madura: padrão Statoil/Equinor pós-1995 (Mar do Norte)
    lambda_acid_base=0.015,
    # Vinculação Lei 12.858 mantida + reforço estatutário hipotético: 50% das
    # 'outras receitas' vinculadas a infra+saúde (50/50). Reforça Estado-aprendiz.
    vinculacao_lei_12858=True,
    vinculacao_outras_receitas=0.5,
    # Aposta institucional: C_inst inicial ×5 (~0.30, Norway-like ao invés de
    # ~0.05 baseline MA). Modela ponto de partida com Estado mais maduro.
    c_inst_inicial_multiplier=5.0,
    # Política de regularização fundiária ativa: dobra decaimento de pressão
    # acumulada e reduz acumulação de novos choques (proteção comunitária).
    delta_pressao_decay=0.06,
    lambda_pressao_acum=0.20,
)


PRESETS: dict[str, Scenario] = {
    "pessimista": SCENARIO_PESSIMISTA,
    "referencia": SCENARIO_REFERENCIA,
    "otimista": SCENARIO_OTIMISTA,
    "sem_lei12858": SCENARIO_SEM_LEI12858,
    "choque_brent": SCENARIO_CHOQUE_BRENT,
    "ma_prospero": SCENARIO_MA_PROSPERO,
}


def by_name(nome: str) -> Scenario:
    """Recupera um preset por nome."""
    if nome not in PRESETS:
        raise KeyError(f"Cenário '{nome}' desconhecido. Disponíveis: {sorted(PRESETS.keys())}")
    return PRESETS[nome]
