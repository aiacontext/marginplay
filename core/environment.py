"""Interface multi-agente do simulador (spec §3 e §4).

Padrão PettingZoo-like: ``reset()`` devolve um dict de observações por agente;
``step(actions)`` aplica ``World.step()`` e devolve dicts de obs/reward/done.
Compatível com qualquer treinador MADDPG que consuma esse formato.

Dimensões alinhadas com spec §4.2 (estendido para zonas §2.6):
- gov_estadual: 17 obs / 5 act
- operadora:    6 obs / 2 act
- regulador:    5 obs / 2 act
- comunidade:  11 obs / 2 act
- gov_federal:  6 obs / 2 act
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core import physics as ph
from core.scenarios import Scenario, by_name
from core.state import N_ZONES, ZONES, WorldState
from core.world import Actions, StepInfo, World, random_actions

# IDs canônicos dos agentes — usados como chave em observations/rewards/actions
# v2 (caso FZA-M-59 2023-2025 prova conflito ANP × IBAMA): regulador unitário
# substituído por 2 agentes com mandatos divergentes.
AGENT_GOV = "gov_estadual"
AGENT_OPER = "operadora"
AGENT_ANP = "anp"  # v2 (Lei 9.478/97): técnico-econômico, autoriza E&P
AGENT_IBAMA = "ibama"  # v2 (Lei 6.938/81 PNMA): licenciamento + fiscalização ambiental
AGENT_COM = "comunidade"
AGENT_FED = "gov_federal"

AGENTS: tuple[str, ...] = (
    AGENT_GOV,
    AGENT_OPER,
    AGENT_ANP,
    AGENT_IBAMA,
    AGENT_COM,
    AGENT_FED,
)

# Dimensões espaços (spec §4.2 v2)
OBS_DIMS: dict[str, int] = {
    AGENT_GOV: 22,  # v2: 14 escalares + 4 K_pub + 4 N_rel (§2.3.4-§2.3.6 + RCL)
    AGENT_OPER: 6,
    AGENT_ANP: 6,  # v2: técnica/econômica — produção, reservas, royalty, conteúdo local
    AGENT_IBAMA: 6,  # v2: ambiental — E_amb, dano, mobilização, accountability
    AGENT_COM: 12,  # v2: +1 K_saude (§2.3.6)
    AGENT_FED: 6,
}
ACT_DIMS: dict[str, int] = {
    AGENT_GOV: 6,  # v2: +1 vs v1 — frac_saude_livre (Lei 12.858 + K_saude novo)
    AGENT_OPER: 2,
    AGENT_ANP: 2,  # v2: ritmo_aprov_PD ∈ [0,1], rigor_seg_op ∈ [0,1]
    AGENT_IBAMA: 2,  # v2: phi_fisc_amb ∈ [0,1], exigencia_compensacao ∈ [0,1]
    AGENT_COM: 2,
    AGENT_FED: 3,  # v2: ritmo_leiloes, alpha_cide, fisc_amb (substitui frac_repasse)
}

# ---------------------------------------------------------------------------
# Parâmetros das funções de utilidade (§3.2 e §3.3 revisados — formas
# côncavas/convexas baseadas em consenso disciplinar).
# ---------------------------------------------------------------------------
# Educação: Cobb-Douglas (Glomm & Ravikumar 1992, JPE).
# Range empírico β ∈ [0.2, 0.6]; midpoint 0.4.
BETA_EDU: float = 0.4

# Fundo soberano: Permanent Income Hypothesis (van der Ploeg & Venables 2011, EJ).
# Norway Oil Fund usa taxa real de longo prazo ~3-4%.
R_PERM_FUNDO: float = 0.04
SHARE_ROY_ANUIDADE: float = 0.1  # parcela do royalty corrente como anuidade implícita

# Operadora — segurança offshore Brasil pós-pré-sal. Calibração com fontes
# brasileiras primárias (RASO/ANP, Petrobras Sustentabilidade, P-36, Frade).
# Internacional (Macondo) só como cauda extrema do range.
GAMMA_SEG: float = 1.4
"""Decaimento exponencial de prob_acidente em α_seg. Calibrado de Petrobras
TAR 2018-2021 (queda de 1.06→0.54 em ~3 anos com gasto SMS dobrado ⇒ γ≈1.7
quando ajustado para escala α_seg ∈ [0,1]). Range empírico [0.9, 2.3]
(IOGP Process Safety 2023, CSB Macondo Report 2016 no extremo)."""

P_ACID_BASELINE: float = 0.12
"""Probabilidade base anual de acidente Tier 1 por instalação. Calibrado
de RASO/ANP 2022 (62 incidentes em ~60 plataformas) e Petrobras
Sustentabilidade 2021 (7 Tier 1/60 instalações = 0.12). Range observado
[0.06, 0.35] (anos limpos vs pico 2023)."""

L_MULTIPLIER: float = 1.8
"""Custo total de acidente / receita do período (mediana brasileira).
P-36 (2001): R$ 2.5-3 bi vs receita Petrobras 2001 R$ 60 bi → L/receita_anual
~0.04. Frade/Chevron (2011-13): TAC R$ 311 mi + multas + suspensão ~0.3-0.5.
Macondo (2010, BP): 7.5× receita (cauda extrema). Mediana brasileira: 1.8."""

C_SEG_UNIT: float = 0.005
"""Custo unitário quadrático em α_seg. Calibrado para α_seg=1 ≈ US$1.8/bbl
em SMS (~3% receita), consistente com Petrobras lifting cost US$5/bbl ×
6-8% SMS (Sindipetro NF; V SOMA 2017 painel 1)."""

RISCO_REG_COEF: float = 0.5
"""Coeficiente do componente quadrático contínuo de risco regulatório
em E_amb. Calibrado para que E_amb=1 gere risco esperado ~R$ 0.5 bi/step
(faixa Frade-TAC). Componente STEP adicionado quando E_amb cruza 0.7
(cliff regulatório, caso Frade-Chevron 2011: suspensão judicial completa
quando passivo cruza limite)."""

E_AMB_CLIFF_THRESHOLD: float = 0.70
"""Limite onde o regulador brasileiro tipicamente impõe suspensão de
operações (precedente Frade-Chevron 2011, ordem judicial)."""

E_AMB_CLIFF_PENALTY: float = 2.0
"""Penalidade de step (R$ bi) quando E_amb > threshold — captura
suspensão judicial / cliff regulatório (Lei 9.605/98 art. 76)."""


@dataclass
class StepResult:
    """Saída de ``MarginPlayEnv.step``."""

    observations: dict[str, np.ndarray]
    rewards: dict[str, float]
    done: bool
    info: StepInfo
    actions_applied: Actions = field(default_factory=dict)


class MarginPlayEnv:
    """Ambiente multi-agente para MADDPG/PPO/etc.

    Sequência por step:
    1. Recebe ``actions`` (dict por agente).
    2. Chama ``world.step(actions)`` (atualiza estado + retorna StepInfo).
    3. Computa observações pós-step.
    4. Computa recompensas usando StepInfo + estado anterior.
    """

    def __init__(
        self,
        uf: str = "MA",
        scenario: Scenario | str = "referencia",
        rng_seed: int = 0,
    ) -> None:
        self.uf = uf
        self.scenario = by_name(scenario) if isinstance(scenario, str) else scenario
        self._rng_seed = rng_seed
        self.world: World | None = None
        self._prev_state_snapshot: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, np.ndarray]:
        """Recria o World do zero e devolve observações iniciais."""
        self.world = World.from_uf(self.uf, self.scenario, rng_seed=self._rng_seed)
        self._prev_state_snapshot = self._snapshot(self.world.state)
        return self._observe(self.world.state)

    def step(self, actions: Actions) -> StepResult:
        """Aplica ações, retorna observações + recompensas + done."""
        if self.world is None:
            raise RuntimeError("Chame reset() antes de step()")

        prev = self._prev_state_snapshot or self._snapshot(self.world.state)
        info = self.world.step(actions)
        rewards = self._rewards(prev, info, actions)
        obs = self._observe(self.world.state)
        done = self.world.state.t >= self.world.horizon_steps
        self._prev_state_snapshot = self._snapshot(self.world.state)
        return StepResult(
            observations=obs,
            rewards=rewards,
            done=done,
            info=info,
            actions_applied=actions,
        )

    # ------------------------------------------------------------------
    # Observações por agente (spec §3.2-3.6 + extensão zonal §4.2)
    # ------------------------------------------------------------------
    def _observe(self, s: WorldState) -> dict[str, np.ndarray]:
        n_total = max(s.N_total, 1e-9)
        ciclo_eleitoral = float((s.t * 2) % 4 == 0)
        n_rel = s.N / n_total
        return {
            AGENT_GOV: np.concatenate(
                [
                    np.array(
                        [
                            s.avg_pop(s.K_pub),
                            s.avg_pop(s.K_hum),
                            s.avg_pop(s.K_saude),  # v2 §2.3.6
                            n_total / 1000.0,
                            s.avg_pop(s.C_inst),
                            s.gini,
                            s.roy_periodo / 1e9,
                            s.W,
                            s.t / 15.0,
                            ciclo_eleitoral,
                            s.fundo_soberano / 1e10,
                            s.pib_estadual / 100.0,  # v2 §2.3.4 — escala R$ centenas bi
                            s.icms_periodo / 10.0,  # v2 §2.3.5
                            s.fpe_periodo / 10.0,  # v2 §2.3.5
                        ]
                    ),
                    s.K_pub.astype(float),
                    n_rel.astype(float),
                ]
            ),
            AGENT_OPER: np.array(
                [
                    s.R / max(self.scenario.URR_Gbbl, 1e-9),
                    s.P_efetiva,
                    s.preco / 100.0,
                    s.E_amb,
                    self._last_phi_fisc(),
                    s.t / 15.0,
                ]
            ),
            # v2 — ANP (técnico-econômico, vinculada ao MME, Lei 9.478/97)
            AGENT_ANP: np.array(
                [
                    s.P_efetiva,
                    s.R / max(self.scenario.URR_Gbbl, 1e-9),
                    s.preco / 100.0,
                    s.roy_periodo / 1e9,
                    self._last_alpha_seg(),
                    self._pressao_politica(s),
                ]
            ),
            # v2 — IBAMA (ambiental, vinculada ao MMA, Lei 6.938/81 PNMA)
            AGENT_IBAMA: np.array(
                [
                    s.E_amb,
                    s.P_efetiva,
                    self._last_alpha_seg(),
                    self._dano_economico(s),
                    self._pressao_politica(s),
                    self._mobilizacao_proxy(),  # IBAMA responde à mobilização social
                ]
            ),
            AGENT_COM: np.concatenate(
                [
                    np.array(
                        [
                            self._renda_pc_uf(s),
                            s.avg_pop(s.K_pub),
                            s.avg_pop(s.K_hum),
                            s.avg_pop(s.K_saude),  # v2 §2.3.6
                            s.E_amb,
                            n_total / 1000.0,
                            self._emprego_formal_proxy(s),
                            s.gini,
                        ]
                    ),
                    n_rel.astype(float),
                ]
            ),
            AGENT_FED: np.array(
                [
                    s.P_efetiva,
                    s.roy_periodo / 1e9,
                    s.W,
                    s.E_amb,
                    s.preco / 100.0,
                    s.t / 15.0,
                ]
            ),
        }

    # ------------------------------------------------------------------
    # Recompensas por agente (spec §3.2-3.6)
    # ------------------------------------------------------------------
    def _rewards(
        self,
        prev: dict[str, Any],
        info: StepInfo,
        actions: Actions,
    ) -> dict[str, float]:
        s = self.world.state  # type: ignore[union-attr]

        # Ações do step (usadas para incentivos diretos sem aguardar lag).
        # v4: gov tem 6 dims [edu_livre, saude_livre, infra, inst, fundo, interior].
        gov_act = np.asarray(actions.get(AGENT_GOV, np.zeros(6)), dtype=float)
        oper_act = np.asarray(actions.get(AGENT_OPER, np.zeros(2)), dtype=float)
        frac_edu_livre = float(gov_act[0]) if gov_act.size >= 1 else 0.0
        frac_infra = float(gov_act[2]) if gov_act.size >= 3 else 0.0
        frac_inst = float(gov_act[3]) if gov_act.size >= 4 else 0.0
        # frac_interior (gov_act[5]) entra na física via _allocate_zonal, não na reward direta.
        frac_saude_livre = float(gov_act[1]) if gov_act.size >= 2 else 0.0
        alpha_seg = float(oper_act[1]) if oper_act.size >= 2 else 0.0

        # ΔW
        delta_w = float(info.W - prev["W"])
        ciclo_eleitoral = float((s.t * 2) % 4 == 0)
        delta_kpub = float(info.avg_K_pub - prev["avg_K_pub"])
        deficit_pen = max(0.0, prev["roy_periodo"] / 1e9 * 0.1 - info.roy_periodo / 1e9)

        # ============================================================
        # Gov Estadual: §3.2 v4 — REFORMULAÇÃO METODOLÓGICA COMPLETA
        # ============================================================
        # Cada categoria recebe FORMA FUNCIONAL ESPECÍFICA conforme literatura:
        # - K_hum/K_saude: CRRA (Glomm-Ravikumar adaptado; sem singularidade Inada)
        # - K_pub: Aschauer-Munnell com saturação (Frischtak-Mourão 2017)
        # - C_inst: corretivo Acemoglu-North (Mendes 2010)
        # - Fundo: PIH log (van der Ploeg 2011)
        # - Equidade zonal: Atkinson 1970 (CF/88 art. 3º)
        # - Visibilidade: estendida a saúde+educação (Drazen-Eslava 2010)
        # - ΔW: reduzido (era 0.5, agora 0.10) — evita dupla contagem
        rcl_livre_bi = info.icms_periodo + info.fpe_periodo + info.outras_receitas

        # ----- (1) CRRA para edu e saúde (substitui Cobb-Douglas) -----
        # u_CRRA(x; η) = ((x+ε)^(1-η) − 1) / (1-η)
        # η=0.5 (concavidade moderada); ε=1.0 floor IMPORTANTE: limita marginal
        # em x=0 a 1/√ε=1 (vs ~10 com ε=0.01 que ainda causa dominância).
        eps_floor = 1.0
        eta_crra = 0.5
        x_edu = max(frac_edu_livre * rcl_livre_bi, 0.0) + eps_floor
        x_saude = max(frac_saude_livre * rcl_livre_bi, 0.0) + eps_floor
        u_edu = (x_edu ** (1.0 - eta_crra) - 1.0) / (1.0 - eta_crra)
        u_saude = (x_saude ** (1.0 - eta_crra) - 1.0) / (1.0 - eta_crra)

        # ----- (2) Aschauer com saturação para infra (NOVO v4) -----
        # u_pub_direta = θ · ln(1 + I·gap_K) com gap_K = (1 − K_pub/K_target)
        # Captura: investir em infra rende mais quando K_pub é baixo.
        # Calibração: Frischtak-Mourão 2017; β empírico BR ~0.12 (Araújo IPEA Radar 69).
        k_target_pub = 1.0
        gap_kpub = max(0.0, 1.0 - info.avg_K_pub / k_target_pub)
        u_infra = float(np.log1p(max(frac_infra * rcl_livre_bi, 0.0) * gap_kpub))

        # ----- (3) Corretivo institucional (NOVO v4) -----
        # u_inst = √(I·RCL) · (1 − C_inst_avg)^β  com β=1.5
        # Marginal alta quando C_inst baixo; satura quando estado é maduro.
        c_inst_avg = float(s.avg_pop(s.C_inst))
        gap_cinst = max(0.0, 1.0 - c_inst_avg)
        u_inst = float(np.sqrt(max(frac_inst * rcl_livre_bi, 0.0))) * (gap_cinst ** 1.5)

        # ----- (4) PIH para fundo soberano (mantido v3) -----
        c_perm = R_PERM_FUNDO * info.fundo_soberano + SHARE_ROY_ANUIDADE * info.roy_periodo
        bonus_fundo = float(np.log(max(c_perm / 1e9, 1e-3)))

        # ----- (5) Equidade zonal Atkinson (NOVO v4, ε=1 caso log) -----
        # W̃_atkinson = exp(mean_z log(renda_z)) — média geométrica zonal.
        # Captura aversão à desigualdade espacial (CF/88 art. 3º).
        renda_pc_zone = self._renda_pc_zone(s)
        log_rendas = np.log(np.maximum(renda_pc_zone, 100.0))
        w_atkinson = float(np.exp(log_rendas.mean()))
        u_equity = float(np.log(max(w_atkinson / 1000.0, 1e-3)))  # escala ~milhares R$

        # ----- (6) Visibilidade ESTENDIDA (Drazen-Eslava 2010) -----
        # Inclui ΔK_saude e ΔK_hum (Mais Médicos, vagas escolares são visíveis).
        delta_ksaude = float(info.avg_K_saude - prev.get("K_saude_avg", info.avg_K_saude))
        delta_khum = float(info.avg_K_hum - prev.get("K_hum_avg", info.avg_K_hum))
        delta_visivel = 0.5 * delta_kpub + 0.3 * delta_ksaude + 0.2 * delta_khum
        visib = float(np.log1p(max(0.0, delta_visivel * ciclo_eleitoral)))

        # ----- Composição final R_gov v4 (soma ponderada — Persson-Tabellini) -----
        # Pesos rebalanceados após análise de marginais: edu/saúde têm CRRA
        # com floor=1, mas ainda dominam em frac=0; aumentei pesos de
        # infra/inst para compensar e induzir diversificação realista.
        r_gov = (
            0.10 * delta_w
            + 0.12 * u_edu
            + 0.12 * u_saude
            + 0.20 * u_infra
            + 0.10 * u_inst
            + 0.07 * bonus_fundo
            + 0.10 * u_equity
            + 0.20 * visib
            - 0.10 * deficit_pen
        )

        # Operadora: §3.3 (revisado v1.1) — Viscusi 1983 + Bier 2004 + Macondo.
        # Custo quadrático (convexo) + probabilidade exponencial-decrescente em α_seg
        # + penalidade de acidente como custo ESPERADO (não evento Bernoulli).
        receita = info.P_efetiva * info.preco * (1 - self.scenario.tau_royalty) * 1e-3
        # Lifting cost pré-sal Petrobras ~US$6.50/bbl (Form 20-F 2024 SEC),
        # Brent ~US$80/bbl → ~8% receita bruta. Calibrado em 0.08·P·preço·1e-3
        # (substitui placeholder 0.001·P que ignorava preço).
        custo_op = 0.08 * info.P_efetiva * info.preco * 1e-3
        custo_seg = C_SEG_UNIT * (alpha_seg**2) * info.P_efetiva  # quadrático
        custo_reg = 0.01 * self._last_phi_fisc() * info.P_efetiva
        # Custo esperado de acidente: prob × loss (Viscusi). Loss = L_multiplier × receita.
        p_acid_esp = P_ACID_BASELINE * float(np.exp(-GAMMA_SEG * alpha_seg))
        pen_acid = p_acid_esp * L_MULTIPLIER * receita
        # Risco regulatório: quadrático contínuo (Shavell 1984) + componente
        # STEP quando E_amb > threshold (cliff Frade-Chevron 2011).
        risco_reg_quad = RISCO_REG_COEF * (s.E_amb**2)
        risco_reg_step = E_AMB_CLIFF_PENALTY if s.E_amb > E_AMB_CLIFF_THRESHOLD else 0.0
        risco_reg = risco_reg_quad + risco_reg_step
        r_oper = receita - custo_op - custo_seg - custo_reg - pen_acid - risco_reg

        # Reguladores v2 — split em 2 agentes (caso FZA-M-59 2023-2025).
        # ANP (Lei 9.478/97) — vinculada ao MME, pró-receita estatal e
        # segurança operacional. Calibração: TCU Acórdão 2.936/2021.
        # α_arrec=0.10 calibrado para equilíbrio Nash (sweep 10k v2 mostrou
        # que 0.50 fazia ANP dominar escala do sistema no otimista — Q≈800,
        # 6× maior que gov_federal). ANP recebe receita via taxa de
        # fiscalização (~0.3% faturamento) + bônus de assinatura, não o
        # royalty inteiro — peso 0.10 reflete interesse institucional sem
        # capturar receita que constitucionalmente pertence à União/estados.
        delta_e_amb = s.E_amb - prev["E_amb"]
        dano_econ = self._dano_economico(s)
        pressao = self._pressao_politica(s)
        receita_estatal_proxy = info.roy_periodo / 1e9
        r_anp = (
            +0.10 * receita_estatal_proxy  # arrecadação institucional ANP
            - 0.20 * delta_e_amb  # peso ambiental MENOR (mandato técnico)
            - 0.15 * dano_econ
            - 0.15 * (self._last_phi_fisc() * pressao)
        )

        # IBAMA (Lei 6.938/81 PNMA) — vinculada ao MMA, pró-procedimento
        # ambiental. Calibração: dataset IBAMA fiscalizacao-auto-de-infracao
        # + Acórdão TCU 2.936/2021 (mandato procedimental, não outcome).
        accountability = 1.0 - float(np.clip(s.E_amb - 0.3, 0.0, 0.7))  # floor procedural
        r_ibama = (
            -0.55 * delta_e_amb  # peso ambiental DOMINANTE
            - 0.30 * dano_econ
            + 0.15 * accountability  # mandato estatutário (TCU 2021)
            - 0.10 * (self._last_phi_fisc() * pressao)  # captura mais fraca que ANP
        )

        # Comunidade: §3.5 v2 — pesos revisados de Escobar (2008), Almeida
        # (NAEA/UFPA 2008), CIMI 2023. Comunidades amazônicas valorizam
        # menos renda monetária, mais ambiente e território.
        com_act = np.asarray(actions.get(AGENT_COM, np.zeros(2)), dtype=float)
        mobilizacao = float(com_act[0]) if com_act.size >= 1 else 0.0
        # pref_amb (com_act[1]) entra na física via aplica_mobilizacao em world.step
        renda_pc_zone = self._renda_pc_zone(s)
        prev_renda_zone = prev["renda_pc_zone"]
        r_com_local = 0.0
        n_total = max(s.N_total, 1e-9)
        # Proxy de TERRITÓRIO (estabilidade territorial) — H-TERR-2 (calibrado).
        # Forma logística com histerese, calibrada com CPT 2020-2024, INCRA-MA
        # (gargalo titulação 0.7%), CIMI 2024 (49 invasões TI MA), Andersen
        # et al. (2022), Watts (2004), Escobar (2008), Almeida (NAEA 2008).
        # Vetor zonal — cada mesorregião tem proxy próprio.
        territorio_proxy_zone = ph.compute_territorio_proxy_zonal(
            e_amb=s.E_amb,
            rent_per_capita_zone=info.rent_per_cap_zone,
            c_inst_zone=s.C_inst,
            s_pressao_zone=s.S_pressao_terr,
        )
        for z in range(N_ZONES):
            d_log_renda = float(
                np.log(max(renda_pc_zone[z], 1.0)) - np.log(max(prev_renda_zone[z], 1.0))
            )
            d_kpub = float(s.K_pub[z] - prev["K_pub"][z])
            d_ksaude = float(s.K_saude[z] - prev.get("K_saude", s.K_saude)[z])
            d_n_rel = float((s.N[z] - prev["N"][z]) / max(prev["N"][z], 1.0))
            # Pesos v2: renda 0.20 (↓ de 0.30), K_pub 0.15, K_saude 0.20,
            # ambiente 0.25 (↑ de 0.20), TERRITORIO 0.10 (NOVO), congestão 0.10
            r_z = (
                0.20 * d_log_renda
                + 0.15 * d_kpub
                + 0.20 * d_ksaude
                + 0.25 * (1.0 - s.E_amb)
                + 0.10 * float(territorio_proxy_zone[z])
                - 0.10 * max(0.0, d_n_rel - 0.05)
            )
            r_com_local += (s.N[z] / n_total) * r_z
        # Custo de mobilização (Olson + Tarrow + CPT 2023). mob_hist ~0.5 placeholder
        # até implementarmos o buffer histórico (similar a I_educ_lag).
        custo_mob = 0.08 * (mobilizacao**2 + 0.5 * mobilizacao * 0.5)
        r_com = r_com_local - 0.15 * s.gini - custo_mob

        # Gov Federal: §3.6 v2 — agente único com função composta multi-mandato.
        # Pesos calibrados de IBGE (PIB petróleo ~13% PIB BR), STF/ADI 4917
        # (Lei 9.478/97 art. 49, ~22% União), MMA NDC 2024, EPE/PDE 2034.
        # Ações: ritmo_leiloes (afeta URR futura), alpha_cide (CIDE-Combustíveis),
        # fisc_amb (orçamento IBAMA/ICMBio).
        gov_fed_act = np.asarray(actions.get(AGENT_FED, np.zeros(3)), dtype=float)
        # ritmo_leiloes (gov_fed_act[0]) entra na física via Hubbert; aqui só
        # contribui para receita_uniao via alpha_cide e custo_fisc_amb.
        alpha_cide_fed = float(gov_fed_act[1]) if gov_fed_act.size >= 2 else 0.5
        fisc_amb_fed = float(gov_fed_act[2]) if gov_fed_act.size >= 3 else 0.5

        # Distribuição estatutária Lei 9.478/97 art. 49 (parcela União ~22%
        # offshore concessão pós-12.734/12). Substitui o `·0.3` ad-hoc.
        roy_federal_bi = info.roy_periodo / 1e9 * 0.22
        # CIDE: receita federal adicional proporcional a alpha_cide × produção
        cide_receita = alpha_cide_fed * info.P_efetiva * info.preco * 1e-3 * 0.05
        receita_uniao = roy_federal_bi + cide_receita

        # Segurança energética: razão de auto-suficiência (proxy via produção)
        seg_energ = float(np.tanh(info.P_efetiva / 1.0))  # satura em 1 quando P_efetiva alta

        # Custo do gasto em fiscalização ambiental (orçamento federal).
        # fisc_amb_fed agora reduz alpha_prod em update_E_amb (world.step),
        # fechando o loop físico — aqui resta apenas o custo orçamentário.
        custo_fisc_amb = 0.5 * fisc_amb_fed

        # Termo ambiental consolidado (substitui double-counting v2):
        # v2 tinha -0.25·E_amb + 0.05·(1 − E_amb) = -0.30·E_amb + 0.05.
        # v3 usa -0.30·E_amb direto (algebricamente idêntico, mais legível;
        # captura simultaneamente NDC 2024 e soft power COP30 Belém).
        r_fed = (
            +0.35 * receita_uniao  # arrecadação Tesouro
            + 0.15 * seg_energ  # MME/EPE PDE 2034
            - 0.30 * s.E_amb  # MMA NDC 2024 + Itamaraty COP30 (consolidado v3)
            + 0.20 * info.W  # MDR — equidade regional
            - 0.05 * custo_fisc_amb  # custo orçamentário
        )

        return {
            AGENT_GOV: r_gov,
            AGENT_OPER: r_oper,
            AGENT_ANP: r_anp,
            AGENT_IBAMA: r_ibama,
            AGENT_COM: r_com,
            AGENT_FED: r_fed,
        }

    # ------------------------------------------------------------------
    # Snapshot / helpers
    # ------------------------------------------------------------------
    def _snapshot(self, s: WorldState) -> dict[str, Any]:
        return {
            "W": s.W,
            "E_amb": s.E_amb,
            "roy_periodo": s.roy_periodo,
            "avg_K_pub": s.avg_pop(s.K_pub),
            "K_pub": s.K_pub.copy(),
            "K_saude": s.K_saude.copy(),
            "K_saude_avg": s.avg_pop(s.K_saude),  # v4 — para visibilidade estendida
            "K_hum_avg": s.avg_pop(s.K_hum),  # v4 — para visibilidade estendida
            "N": s.N.copy(),
            "renda_pc_zone": self._renda_pc_zone(s).copy(),
        }

    def _last_phi_fisc(self) -> float:
        # Sem histórico de ações: assume valor neutro
        return 0.5

    def _last_alpha_seg(self) -> float:
        return 0.5

    def _mobilizacao_proxy(self) -> float:
        """Proxy para nível de mobilização da comunidade (v2 §3.5)."""
        return 0.5  # placeholder — será atualizado quando _com afetar física

    def _dano_economico(self, s: WorldState) -> float:
        """Dano = δ_pesca * E_amb + δ_turismo * E_amb² (spec §2.2.6)."""
        return 0.3 * s.E_amb + 0.5 * s.E_amb**2

    def _pressao_politica(self, s: WorldState) -> float:
        """Roy / PIB estadual proxy."""
        pib_uf = max(s.N_total * 18.0, 1.0)  # mil R$/hab
        return float(np.clip((s.roy_periodo / 1e6) / pib_uf, 0.0, 1.0))

    def _renda_pc_uf(self, s: WorldState) -> float:
        return float(s.avg_pop(self._renda_pc_zone(s)))

    def _renda_pc_zone(self, s: WorldState) -> np.ndarray:
        # Renda base + bônus de royalties zonais (proxy)
        if s.roy_periodo <= 0:
            return np.full(N_ZONES, 18000.0)
        share_costa = 0.5
        share_int = 0.5
        n_int = max(s.N[1:].sum(), 1.0)
        bonus = np.zeros(N_ZONES)
        bonus[0] = s.roy_periodo * share_costa / max(s.N[0] * 1000.0, 1.0)
        bonus[1:] = (
            s.roy_periodo * share_int * (s.N[1:] / n_int) / np.maximum(s.N[1:] * 1000.0, 1.0)
        )
        return 18000.0 + bonus

    def _emprego_formal_proxy(self, s: WorldState) -> float:
        return float(np.clip(0.3 + 0.5 * s.avg_pop(s.K_pub) + 0.2 * s.P_efetiva, 0.0, 1.0))


def random_episode(env: MarginPlayEnv, rng_seed: int = 123) -> list[StepResult]:
    """Roda um episódio inteiro com ações aleatórias. Útil para smoke."""
    rng = np.random.default_rng(rng_seed)
    env.reset()
    history: list[StepResult] = []
    while True:
        actions = random_actions(rng)
        result = env.step(actions)
        history.append(result)
        if result.done:
            break
    return history


# Re-exports úteis
__all__ = [
    "ACT_DIMS",
    "AGENTS",
    "AGENT_ANP",
    "AGENT_COM",
    "AGENT_FED",
    "AGENT_GOV",
    "AGENT_IBAMA",
    "AGENT_OPER",
    "OBS_DIMS",
    "ZONES",
    "MarginPlayEnv",
    "StepResult",
    "random_episode",
]
