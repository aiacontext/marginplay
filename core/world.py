"""Motor principal da simulação (spec §2).

``World`` orquestra ``WorldState`` + ``Scenario`` + funções puras de
``physics`` para avançar a simulação um step (2 anos por padrão).

Recebe um dicionário de ações por agente (formato definido em §3 e
``agents/definitions.py``). Para esta primeira versão, um helper
``random_actions`` gera ações válidas para smoke tests sem MADDPG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np

from core import physics as ph
from core import scenarios as sc
from core.scenarios import Scenario
from core.state import N_ZONES, WorldState, load_initial_state


class Actions(TypedDict, total=False):
    """Ações dos 6 agentes para um step (formato spec §3 v2)."""

    # gov_estadual: 6 frações sobre RCL livre
    gov_estadual: np.ndarray
    operadora: np.ndarray  # [alpha_invest, alpha_seg]
    anp: np.ndarray  # [ritmo_aprov_PD, rigor_seg_op]
    ibama: np.ndarray  # [phi_fisc_amb, exigencia_compensacao]
    comunidade: np.ndarray  # [mobilizacao, preferencia_amb]
    gov_federal: np.ndarray  # [ritmo_leiloes, alpha_cide, fisc_amb]


@dataclass
class StepInfo:
    """Métricas e diagnósticos retornados a cada step."""

    t: int
    R_remanescente: float
    P_efetiva: float
    preco: float
    roy_periodo: float
    gini: float
    W: float
    acidente: bool
    severidade_acidente: float
    n_total: float
    avg_K_pub: float  # noqa: N815
    avg_K_hum: float  # noqa: N815
    avg_C_inst: float  # noqa: N815
    fundo_soberano: float = 0.0  # stock acumulado em R$ (spec §2.3.3)
    juros_fundo: float = 0.0  # juros do período em R$ (entram em renda_pc)
    # ----- v2 (modelo realista, spec §2.3.4-§2.3.6) -----
    avg_K_saude: float = 0.0  # noqa: N815 — capital saúde médio populacional
    K_priv: float = 0.0
    pib_estadual: float = 0.0  # PIB estadual (R$ bi)
    icms_periodo: float = 0.0  # R$ bi do período
    fpe_periodo: float = 0.0  # R$ bi do período
    fundeb_periodo: float = 0.0  # R$ bi do período
    # Rent per capita zonal normalizada — exposto para que rewards (R_com)
    # possam computar territorio_proxy zonal via H-TERR-2 sem recalcular.
    rent_per_cap_zone: np.ndarray = field(default_factory=lambda: np.zeros(N_ZONES))
    outras_receitas: float = 0.0  # R$ bi do período
    rcl_total: float = 0.0  # soma de todas as receitas do período (R$ bi)


@dataclass
class World:
    """Mundo simulado: 1 UF, N_ZONES=4 zonas, horizonte de 15 steps."""

    state: WorldState
    scenario: Scenario
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    horizon_steps: int = 15
    dt_anos: float = 2.0
    PIB_BASE_BRL_PER_HAB: float = 18000.0
    R_FUNDO_ANUAL: float = 0.04  # rendimento anual do fundo soberano (~títulos públicos)

    # Pool nacional IR+IPI (R$ bi) — exógeno, alimenta FPE de todas UFs.
    pool_ir_ipi_bi: float = sc.POOL_IR_IPI_INICIAL_BI

    # Histórico opcional para análise (cada step empilha aqui)
    history: list[StepInfo] = field(default_factory=list)

    @classmethod
    def from_uf(
        cls,
        uf: str,
        scenario: Scenario,
        rng_seed: int = 0,
    ) -> World:
        """Construtor conveniente: carrega estado inicial da UF + cenário."""
        state = load_initial_state(
            uf=uf,
            R0_Gbbl=scenario.URR_Gbbl,
            preco0_usd=scenario.preco0_usd,
            c_inst_inicial_multiplier=scenario.c_inst_inicial_multiplier,
        )
        return cls(state=state, scenario=scenario, rng=np.random.default_rng(rng_seed))

    # ------------------------------------------------------------------
    # Step principal
    # ------------------------------------------------------------------
    def step(self, actions: Actions) -> StepInfo:
        """Aplica as transições de §2 v2 (modelo realista) e atualiza o estado.

        Sequência:
        1. Preço (random walk)
        2. Produção (Hubbert × α_invest)
        3. Royalties (com repasse federal)
        4. Receitas estaduais: ICMS endógeno (PIB), FPE (cota fixa), FUNDEB,
           Outras (drift).  Soma = RCL_total.
        5. **Lei 12.858/2013 (rígida na física):** vinculações automáticas
           75% royalties → educação, 25% royalties → saúde, FUNDEB → educação.
        6. RCL_livre = ICMS + FPE + Outras. Agente decide via 6 frações.
        7. Orçamentos finais por setor (vinculados + livres).
        8. Updates de K_pub, K_hum, K_saude, C_inst, K_priv, fundo soberano.
        9. População, acidente, E_amb, PIB estadual, W.
        """
        # Default v2: 6 dims [edu_livre, saude_livre, infra, inst, fundo, interior]
        gov = self._unpack(
            actions,
            "gov_estadual",
            default=np.array([0.20, 0.15, 0.25, 0.10, 0.10, 0.30]),
        )
        oper = self._unpack(actions, "operadora", default=np.array([1.0, 0.5]))
        anp = self._unpack(actions, "anp", default=np.array([0.5, 0.5]))
        ibama = self._unpack(actions, "ibama", default=np.array([0.5, 0.5]))
        com = self._unpack(actions, "comunidade", default=np.array([0.5, 0.5]))
        fed = self._unpack(actions, "gov_federal", default=np.array([0.5, 0.5, 0.5]))
        mobilizacao_com, pref_amb_com = com

        # Suporta v1 (5 dims) e v2 (6 dims) via comprimento da ação.
        if gov.size >= 6:
            frac_edu_livre, frac_saude_livre, frac_infra, frac_inst, frac_fundo, frac_interior = (
                gov[:6]
            )
        else:
            frac_edu_livre, frac_infra, frac_inst, frac_fundo, frac_interior = gov[:5]
            frac_saude_livre = 0.0
        alpha_invest, alpha_seg = oper
        # ANP (Lei 9.478/97 + Res. 882/2022):
        # - ritmo_aprov_PD: aprovação de Planos de Desenvolvimento — gargalo
        #   regulatório que limita capacidade efetiva (TCU 2.936/2021 sobre lag
        #   PD-PD médio de 18 meses na MEB).
        # - rigor_seg_op: exigência de Programa de Segurança Operacional —
        #   reduz prob. de acidente (sinergia com α_seg da operadora).
        ritmo_aprov_pd, rigor_seg_op_anp = anp
        # IBAMA (Lei 6.938/81 PNMA + Lei 9.985/00 art. 36):
        # - phi_fisc: fiscalização ambiental (entra em prob_acidente e E_amb)
        # - exigencia_compensacao: piso de remediação obrigatória (compensação
        #   ambiental SNUC) — força i_amb mínimo independente de α_seg.
        phi_fisc, exig_comp = ibama
        # Gov federal v2: ritmo_leiloes (→ ΔURR), alpha_cide, fisc_amb.
        # fisc_amb_fed: orçamento IBAMA/ICMBio (PPA 2024-2027) — reduz
        # alpha_prod (impacto ambiental por Mbbl/d) na próxima rodada.
        ritmo_leiloes, _alpha_cide, fisc_amb_fed = fed
        # Repasse de royalties agora é estatutário (não decisão); usa default 1.0
        # (parcela federal aplicada em roy_split — ver §2.3.7 v2).
        frac_repasse = 1.0

        s = self.state

        # 1) Preço — random walk + choque exógeno (contrafactual Choque-Brent)
        s.preco = ph.step_preco(
            s.preco,
            sigma_anual=self.scenario.sigma_preco,
            mu_drift_anual=self.scenario.mu_drift,
            dt_anos=self.dt_anos,
            rng=self.rng,
        )
        # Aplica choque de Brent quando configurado (cenário Choque-Brent).
        if (
            self.scenario.price_shock_step is not None
            and s.t == self.scenario.price_shock_step
        ):
            s.preco *= float(self.scenario.price_shock_factor)

        # 2) Produção: Hubbert × ação Operadora × cap por reservas.
        # ritmo_leiloes (União) eleva P_max efetivo via novos blocos arrematados.
        # Calibração: ANP 11ª/16ª Rodada e OPC ciclos 1-5 — blocos arrematados
        # que viram capacidade efetiva em janela de 2 anos (dt do simulador) =
        # 10-20% (lag exploratório de declarações de comercialidade pré-PD).
        # Por isso leilao_factor max = 1.20 (vs v4 inicial 1.30 que extrapolava).
        # ritmo_aprov_PD (ANP) aplica gargalo regulatório: PD não aprovado
        # bloqueia desenvolvimento de campos já arrematados (TCU 2.936/2021,
        # lag PD-PD médio MEB ~18 meses). Range 0.8-1.2 (±20% em torno do baseline).
        # Combinação multiplicativa porque os fatores são complementares
        # (estoque de blocos × fluxo de aprovações), não substitutos.
        leilao_factor = 1.0 + 0.20 * float(np.clip(ritmo_leiloes, 0.0, 1.0))
        anp_aprov_factor = 0.8 + 0.4 * float(np.clip(ritmo_aprov_pd, 0.0, 1.0))
        p_max_efetivo = self.scenario.P_max_mbbld * leilao_factor * anp_aprov_factor
        p_potencial = ph.hubbert_production(
            t_step=s.t,
            P_max_mbbld=p_max_efetivo,
            t_peak_step=self.scenario.t_peak_step,
            a_assimetria=self.scenario.a_assimetria,
        )
        s.P_efetiva = ph.producao_efetiva(p_potencial, alpha_invest, s.R)
        s.R = ph.update_reservas(s.R, s.P_efetiva, self.dt_anos)

        # 3) Royalties — aplica frac_repasse do Gov Federal
        roy_total = ph.royalties_periodo(
            s.P_efetiva, s.preco, self.scenario.tau_royalty, dt_anos=self.dt_anos
        )
        s.roy_periodo = roy_total * float(np.clip(frac_repasse, 0.5, 1.0))
        roy_bi = s.roy_periodo / 1e9  # converte R$ → R$ bi

        # 4) Receitas estaduais (RCL decomposta, spec §2.3.5)
        # 4a) ICMS endógeno ao PIB estadual (que vem do step anterior)
        icms_anual = ph.step_icms(
            pib_estadual_bi=max(s.pib_estadual, 1e-3),
            aliquota_efetiva=sc.ALIQUOTA_ICMS_EFETIVA,
            epsilon=sc.EPSILON_ICMS_PIB,
        )
        s.icms_periodo = icms_anual * self.dt_anos  # R$ bi do período (2 anos)

        # 4b) FPE — cota MA fixa, pool nacional cresce com IR+IPI
        self.pool_ir_ipi_bi = ph.step_pool_ir_ipi(
            self.pool_ir_ipi_bi,
            crescimento_anual=self.scenario.crescimento_ir_ipi_anual,
            dt_anos=self.dt_anos,
        )
        fpe_anual = ph.step_fpe(self.pool_ir_ipi_bi, cota_uf=sc.COTA_FPE_MA)
        s.fpe_periodo = fpe_anual * self.dt_anos  # R$ bi

        # 4c) FUNDEB líquido (semi-determinístico, EC 108/2020)
        s.fundeb_periodo = sc.FUNDEB_LIQUIDO_MA_BI * self.dt_anos

        # 4d) Outras receitas (drift)
        s.outras_receitas = ph.step_outras_receitas(
            max(s.outras_receitas, 5.0),  # piso inicial (R$ bi)
            drift_anual=self.scenario.crescimento_w_nacional,
            dt_anos=self.dt_anos,
        )

        # 5) Lei 12.858/2013 — vinculações OBRIGATÓRIAS (rígida na física).
        # Contrafactual Sem-Lei12858: vinculação OFF, royalties viram RCL livre.
        if self.scenario.vinculacao_lei_12858:
            roy_vinc_educ_bi = sc.SHARE_ROY_EDUC_LEI_12858 * roy_bi
            roy_vinc_saude_bi = sc.SHARE_ROY_SAUDE_LEI_12858 * roy_bi
            roy_para_rcl_livre = 0.0
        else:
            # Revogação simulada: royalties tornam-se RCL livre integral
            roy_vinc_educ_bi = 0.0
            roy_vinc_saude_bi = 0.0
            roy_para_rcl_livre = roy_bi
        # FUNDEB: 100% educação (vinculação constitucional EC 108/2020 — sempre on)
        fundeb_educ_bi = s.fundeb_periodo

        # Cenário transformador MA-Próspero: vincula adicionalmente uma fração
        # das 'outras receitas' a infra+saúde (50/50). Modela reforço estatutário
        # hipotético — política de Estado-aprendiz com dotação fixa para CT&I,
        # SUS e infraestrutura (paralelo a fundos previdenciários estaduais).
        outras_vinc_bi = self.scenario.vinculacao_outras_receitas * s.outras_receitas
        outras_vinc_infra_bi = 0.5 * outras_vinc_bi
        outras_vinc_saude_bi = 0.5 * outras_vinc_bi
        outras_livre_bi = s.outras_receitas - outras_vinc_bi

        # 6) RCL livre = ICMS + FPE + Outras_livre + Royalties (se desvinculados)
        rcl_livre_bi = s.icms_periodo + s.fpe_periodo + outras_livre_bi + roy_para_rcl_livre

        # 7) Orçamentos finais por setor (R$ bi do período)
        # Normaliza frações livres para somar ≤ 1
        soma_alloc = (
            float(frac_edu_livre)
            + float(frac_saude_livre)
            + float(frac_infra)
            + float(frac_inst)
            + float(frac_fundo)
        )
        if soma_alloc > 1.0:
            frac_edu_livre /= soma_alloc
            frac_saude_livre /= soma_alloc
            frac_infra /= soma_alloc
            frac_inst /= soma_alloc
            frac_fundo /= soma_alloc

        # Orçamentos = vinculados (rígidos) + livre (decisão do gov_estadual)
        orc_educ_bi = roy_vinc_educ_bi + fundeb_educ_bi + frac_edu_livre * rcl_livre_bi
        orc_saude_bi = roy_vinc_saude_bi + outras_vinc_saude_bi + frac_saude_livre * rcl_livre_bi
        orc_infra_bi = outras_vinc_infra_bi + frac_infra * rcl_livre_bi
        orc_inst_bi = frac_inst * rcl_livre_bi
        contrib_fundo_bi = frac_fundo * rcl_livre_bi

        # 8) Distribuição zonal
        i_infra_zone, i_educ_zone, i_saude_zone, i_inst_zone = self._allocate_zonal(
            orc_educ_bi=orc_educ_bi,
            orc_saude_bi=orc_saude_bi,
            orc_infra_bi=orc_infra_bi,
            orc_inst_bi=orc_inst_bi,
            frac_interior=float(frac_interior),
        )

        # 9) Fundo soberano: rende sobre saldo + recebe contribuição livre
        crescimento_fundo = (1.0 + self.R_FUNDO_ANUAL) ** self.dt_anos
        juros_fundo = s.fundo_soberano * (crescimento_fundo - 1.0)
        s.fundo_soberano = s.fundo_soberano * crescimento_fundo + contrib_fundo_bi * 1e9

        # 10) Updates de capitais zonais
        s.K_pub = ph.update_K_pub(s.K_pub, i_infra_zone, s.C_inst, dt_anos=self.dt_anos)
        # K_hum: investimento defasado τ=4 steps
        i_educ_lag_zone = s.I_educ_lag[:, 0]
        eps_emigra = self._compute_emigracao(s)
        s.K_hum = ph.update_K_hum(
            s.K_hum, i_educ_lag_zone, s.C_inst, eps_emigra, dt_anos=self.dt_anos
        )
        s.I_educ_lag = ph.shift_lag_buffer(s.I_educ_lag, i_educ_zone)
        # K_saude: investimento defasado τ=2 steps
        i_saude_lag_zone = s.I_saude_lag[:, 0]
        s.K_saude = ph.update_K_saude(
            s.K_saude,
            i_saude_lag_zone,
            s.C_inst,
            eps_emigra,
            dt_anos=self.dt_anos,
            delta_saude=sc.DELTA_KSAUDE_ANUAL,
            eta_saude=sc.ETA_KSAUDE,
        )
        s.I_saude_lag = ph.shift_lag_buffer(s.I_saude_lag, i_saude_zone)

        # 11) C_inst: captura por rent. Comunidade mobilizada ATENUA captura
        # (Avritzer 2009 sobre conselhos participativos brasileiros).
        rent_per_cap_zone = self._rent_per_capita_normalizado(s.roy_periodo, float(frac_interior))
        _phi_eff, rent_per_cap_zone = ph.aplica_mobilizacao(
            mobilizacao=float(mobilizacao_com),
            pref_amb=float(pref_amb_com),
            phi_fisc_reg=float(phi_fisc),
            rent_per_capita_zone=rent_per_cap_zone,
            n_zone=s.N,
        )
        s.C_inst = ph.update_C_inst(s.C_inst, i_inst_zone, rent_per_cap_zone, dt_anos=self.dt_anos)

        # 12) K_priv: alimentado por receita líquida da operadora
        receita_op_bi = s.P_efetiva * s.preco * (1 - self.scenario.tau_royalty) * 1e-3
        s.K_priv = ph.update_K_priv(
            s.K_priv,
            receita_op_bi=receita_op_bi * self.dt_anos,
            share_invest_local=sc.SHARE_RECEITA_OP_INVEST_LOCAL,
            delta_kpriv=sc.DELTA_KPRIV_ANUAL,
            dt_anos=self.dt_anos,
        )

        # 13) População
        w_local_zone = self._renda_per_capita_zone(s)
        w_nacional = self.PIB_BASE_BRL_PER_HAB * (
            (1 + self.scenario.crescimento_w_nacional) ** (s.t * self.dt_anos)
        )
        s.N = ph.update_N(s.N, w_local_zone, w_nacional, dt_anos=self.dt_anos)

        # 14) Acidente — depende de α_seg da operadora E phi_fisc EFETIVO
        # (mobilização da comunidade amplifica fiscalização do IBAMA).
        phi_eff_acid, _ = ph.aplica_mobilizacao(
            mobilizacao=float(mobilizacao_com),
            pref_amb=float(pref_amb_com),
            phi_fisc_reg=float(phi_fisc),
            rent_per_capita_zone=np.zeros(N_ZONES),  # não usamos rent aqui
            n_zone=s.N,
        )
        lambda_acid = ph.prob_acidente(
            s.P_efetiva,
            phi_eff_acid,
            self.scenario.lambda_acid_base,
            alpha_seg=alpha_seg,
            rigor_seg_anp=float(rigor_seg_op_anp),
        )
        acidente = bool(self.rng.random() < lambda_acid * self.dt_anos)
        severidade = float(self.rng.lognormal(0.0, 0.5)) * 0.05 if acidente else 0.0
        # Calibração C v2: 20% do α_seg vai para investimento ambiental
        # (vs 30% v1). Petrobras Plano Estratégico — SMS divide-se em
        # ~40% segurança operacional + ~40% saúde ocupacional + ~20% ambiental.
        i_amb_oper = alpha_seg * 0.20
        # IBAMA exigencia_compensacao impõe piso de remediação (Lei 9.985/00
        # art. 36 + Resolução CONAMA 371/2006). Quando IBAMA exige compensação,
        # operadora é OBRIGADA a investir em remediação independente de α_seg.
        # Calibração: 0.30 reflete que compensação plena (~0.5% custo total
        # licenciado) corresponde a ~30% do esforço SMS-ambiental típico.
        i_amb_min = 0.30 * float(np.clip(exig_comp, 0.0, 1.0))
        i_amb = max(i_amb_oper, i_amb_min)
        # fisc_amb federal (IBAMA/ICMBio orçamento) reduz alpha_prod efetivo:
        # mais inspeção → menos impacto evitável por Mbbl/d. Calibração: 40%
        # de redução máxima quando fisc_amb=1 (PPA 2024-2027 Ação 214H).
        alpha_prod_eff = ph.ALPHA_PROD * (1.0 - 0.4 * float(np.clip(fisc_amb_fed, 0.0, 1.0)))
        prev_e_amb = float(s.E_amb)
        s.E_amb = ph.update_E_amb(
            s.E_amb, s.P_efetiva, phi_eff_acid, i_amb, severidade,
            dt_anos=self.dt_anos, alpha_prod=alpha_prod_eff,
        )
        delta_e_amb_step = float(s.E_amb) - prev_e_amb

        # 14b) S_pressao_terr: estoque histerético de pressão territorial (H-TERR-2).
        # Atualização zonal — captura conflitos CPT documentados (path-dependence
        # Almeida 2008, Escobar 2008). Cenário transformador MA-Próspero pode
        # sobrescrever delta_decay e lambda_acum (regularização fundiária ativa).
        s.S_pressao_terr = ph.update_S_pressao(
            s.S_pressao_terr, delta_e_amb_step, rent_per_cap_zone,
            dt_anos=self.dt_anos,
            delta_decay=(
                self.scenario.delta_pressao_decay
                if self.scenario.delta_pressao_decay is not None
                else ph.DELTA_PRESSAO_DECAY
            ),
            lambda_acum=(
                self.scenario.lambda_pressao_acum
                if self.scenario.lambda_pressao_acum is not None
                else ph.LAMBDA_PRESSAO_ACUM
            ),
        )

        # 15) PIB estadual endógeno (Cobb-Douglas) — para próximo step
        s.pib_estadual = ph.compute_pib_estadual(
            k_priv=s.K_priv,
            avg_k_pub=s.avg_pop(s.K_pub),
            avg_k_hum=s.avg_pop(s.K_hum),
            avg_k_saude=s.avg_pop(s.K_saude),
            n_total_milhares=s.N_total,
            a_tfp=sc.A_TFP,
            alpha=sc.ALPHA_PIB_KPRIV,
            beta=sc.BETA_PIB_KPUB,
            gamma=sc.GAMMA_PIB_KHUM,
            delta=sc.DELTA_PIB_KSAUDE,
            theta=sc.THETA_PIB_N,
        )

        # 16) Indicadores agregados — renda_pc inclui juros do fundo
        n_total = max(s.N_total * 1000.0, 1.0)
        juros_per_capita = juros_fundo / n_total
        renda_pc_uf = self._renda_per_capita_uf(s) + juros_per_capita
        s.gini = ph.gini_zonal(w_local_zone, s.N)
        s.W = ph.bem_estar(
            s.K_pub,
            s.K_hum,
            s.N,
            renda_pc_uf,
            s.E_amb,
            s.gini,
            k_saude_zone=s.K_saude,
            weights=sc.W_WEIGHTS_V2,
        )

        # 17) Avança o tempo
        s.t += 1

        rcl_total_bi = (
            s.icms_periodo + s.fpe_periodo + s.fundeb_periodo + s.outras_receitas + roy_bi
        )
        info = StepInfo(
            t=s.t,
            R_remanescente=s.R,
            P_efetiva=s.P_efetiva,
            preco=s.preco,
            roy_periodo=s.roy_periodo,
            gini=s.gini,
            W=s.W,
            acidente=acidente,
            severidade_acidente=severidade,
            n_total=s.N_total,
            avg_K_pub=s.avg_pop(s.K_pub),
            avg_K_hum=s.avg_pop(s.K_hum),
            avg_C_inst=s.avg_pop(s.C_inst),
            fundo_soberano=s.fundo_soberano,
            juros_fundo=juros_fundo,
            avg_K_saude=s.avg_pop(s.K_saude),
            K_priv=s.K_priv,
            pib_estadual=s.pib_estadual,
            icms_periodo=s.icms_periodo,
            fpe_periodo=s.fpe_periodo,
            fundeb_periodo=s.fundeb_periodo,
            outras_receitas=s.outras_receitas,
            rcl_total=rcl_total_bi,
            rent_per_cap_zone=rent_per_cap_zone.copy(),
        )
        self.history.append(info)
        return info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack(actions: Actions, key: str, default: np.ndarray) -> np.ndarray:
        return np.asarray(actions.get(key, default), dtype=float)

    def _allocate_zonal(
        self,
        orc_educ_bi: float,
        orc_saude_bi: float,
        orc_infra_bi: float,
        orc_inst_bi: float,
        frac_interior: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Distribui orçamentos setoriais (em R$ bi) pelas 4 zonas.

        Spec §2.6: zona produtora z0 recebe (1-frac_interior); demais zonas
        dividem frac_interior por população. Retorna intensidades zonais
        [0, 1] a serem consumidas por update_K_pub/K_hum/K_saude/C_inst.

        Normalização: PIB-base zonal escalado para que orçamentos típicos
        (% do PIB local) virem intensidades capazes de superar depreciação.
        """
        s = self.state
        n = s.N

        share_costa = 1.0 - float(np.clip(frac_interior, 0.0, 1.0))
        share_interior = float(np.clip(frac_interior, 0.0, 1.0))
        n_interior = max(n[1:].sum(), 1e-9)

        def split_bi(orc_bi: float) -> np.ndarray:
            """Split de orçamento (R$ bi) entre zonas; retorna R$ por zona."""
            arr = np.zeros(N_ZONES)
            arr[0] = orc_bi * share_costa * 1e9  # R$
            arr[1:] = orc_bi * share_interior * 1e9 * (n[1:] / n_interior)
            return arr

        # PIB-base zonal em R$ (proxy para escalar)
        pib_base_zone_brl = n * self.PIB_BASE_BRL_PER_HAB
        pib_base_zone_brl = np.maximum(pib_base_zone_brl, 1e6)

        # Intensidade [0,1] = (orçamento_zona / PIB_zona) × 10 (escala spec §2.6)
        i_infra = np.minimum(10.0 * split_bi(orc_infra_bi) / pib_base_zone_brl, 1.0)
        i_educ = np.minimum(10.0 * split_bi(orc_educ_bi) / pib_base_zone_brl, 1.0)
        i_saude = np.minimum(10.0 * split_bi(orc_saude_bi) / pib_base_zone_brl, 1.0)
        i_inst = np.minimum(10.0 * split_bi(orc_inst_bi) / pib_base_zone_brl, 1.0)
        return i_infra, i_educ, i_saude, i_inst

    def _rent_per_capita_normalizado(self, roy_periodo: float, frac_interior: float) -> np.ndarray:
        """Rent per capita por zona, escalonado para [0,1] do termo de captura."""
        s = self.state
        share_interior = float(np.clip(frac_interior, 0.0, 1.0))
        share_costa = 1.0 - share_interior
        n_interior = max(s.N[1:].sum(), 1e-9)
        rent = np.zeros(N_ZONES)
        rent[0] = roy_periodo * share_costa / max(s.N[0], 1.0)
        rent[1:] = roy_periodo * share_interior * (s.N[1:] / n_interior) / np.maximum(s.N[1:], 1.0)
        # Normalização: rent_per_cap em milhares R$/hab → fração [0,1]
        return np.clip(rent / 5000.0, 0.0, 1.0)

    def _renda_per_capita_zone(self, s: WorldState) -> np.ndarray:
        """Renda pc por zona = PIB base + roy_zonal/N_zona."""
        roy_zone_share = self._rent_per_capita_normalizado(s.roy_periodo, frac_interior=0.5)
        # roy_zone_share está normalizado; reverter para R$
        return self.PIB_BASE_BRL_PER_HAB + roy_zone_share * 5000.0

    def _renda_per_capita_uf(self, s: WorldState) -> float:
        """Renda pc UF = avg_pop dos zonais."""
        return float(s.avg_pop(self._renda_per_capita_zone(s)))

    def _compute_emigracao(self, s: WorldState) -> np.ndarray:
        """ε_emigra: proporcional ao gap de renda zona vs costa produtora."""
        renda = self._renda_per_capita_zone(s)
        ref = max(renda[0], 1.0)  # costa produtora como referência
        gap = np.clip(1.0 - renda / ref, 0.0, 0.5)
        return gap * 0.02  # max 1% emigração extra por step


def random_actions(rng: np.random.Generator) -> Actions:
    """Gera ações aleatórias válidas para smoke testing (6 agentes v2)."""
    # gov_estadual: dirichlet 5 frações (edu_liv, saude_liv, infra, inst, fundo)
    # + sigmoid frac_interior
    alloc = rng.dirichlet(np.ones(5))
    frac_interior = float(rng.uniform(0.1, 0.7))
    return {
        "gov_estadual": np.array([*alloc, frac_interior]),
        "operadora": np.array([rng.uniform(0.5, 1.2), rng.uniform(0.0, 1.0)]),
        "anp": np.array([rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]),
        "ibama": np.array([rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]),
        "comunidade": np.array([rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]),
        "gov_federal": np.array(
            [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
        ),
    }
