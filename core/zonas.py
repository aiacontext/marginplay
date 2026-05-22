"""Classificação zonal da Margem Equatorial (spec §2.6).

Cada UF-alvo (MA, AP, PA, RN) é particionada em quatro zonas:

- ``COSTA_PRODUTORA``       — litorâneo + produção ativa ou bloco licitado
- ``COSTA_NAO_PRODUTORA``   — litorâneo sem envolvimento direto com petróleo
- ``INTERIOR_MEDIO_ALTO``   — não-litorâneo, IDHM/PIB pc ≥ mediana da UF
- ``INTERIOR_BAIXO``        — não-litorâneo, IDHM/PIB pc < mediana da UF

A classificação é pragmática e reproduzível:

1. **Costeiro**: pertence à lista ``COSTEIROS`` (curada via IBGE Perfil
   dos Municípios Costeiros 2012 + Decreto 5.300/2004 PNGC-II). Códigos
   verificados contra IBGE servicodados em 2026-04.
2. **Produtor**: pertence à lista ``PRODUTORES_ATUAIS`` (ANP 2024: bacia
   Potiguar onshore/offshore + hubs com blocos licitados da Margem
   Equatorial). Estado inicial; cenários podem promover novos produtores.
3. **Corte IDH**: comparação com mediana da UF sobre ``hdi_proxy`` (default:
   PIB per capita, até Atlas Brasil 2022 estar disponível via Base dos Dados).

Referências: spec §2.6, §7.6; Aragón & Rud (2013); Caselli & Michaels (2013).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Zone(Enum):
    """Categorias zonais do modelo."""

    COSTA_PRODUTORA = "costa_produtora"
    COSTA_NAO_PRODUTORA = "costa_nao_produtora"
    INTERIOR_MEDIO_ALTO = "interior_medio_alto"
    INTERIOR_BAIXO = "interior_baixo"


# ---------------------------------------------------------------------------
# Municípios costeiros (80 total, verificados contra IBGE 2026-04)
# ---------------------------------------------------------------------------
_COSTEIROS_MA: frozenset[int] = frozenset(
    [
        2100204,  # Alcântara
        2100832,  # Apicum-Açu
        2100907,  # Araioses
        2101251,  # Bacabeira
        2101301,  # Bacuri
        2101905,  # Bequimão
        2102606,  # Cândido Mendes
        2102903,  # Carutapera
        2103109,  # Cedral
        2104305,  # Godofredo Viana
        2104909,  # Guimarães
        2105005,  # Humberto de Campos
        2105104,  # Icatu
        2106201,  # Luís Domingues
        2107506,  # Paço do Lumiar
        2108058,  # Paulino Neves
        2109403,  # Primeira Cruz
        2109452,  # Raposa
        2110278,  # Santo Amaro do Maranhão
        2111201,  # São José de Ribamar
        2111300,  # São Luís
        2112407,  # Turiaçu
        2112506,  # Tutóia
    ]
)

_COSTEIROS_AP: frozenset[int] = frozenset(
    [
        1600105,  # Amapá
        1600204,  # Calçoene
        1600212,  # Cutias
        1600253,  # Itaubal
        1600303,  # Macapá
        1600501,  # Oiapoque
        1600550,  # Pracuúba
        1600709,  # Tartarugalzinho
    ]
)

_COSTEIROS_PA: frozenset[int] = frozenset(
    [
        1500305,  # Afuá
        1500701,  # Anajás
        1500909,  # Augusto Corrêa
        1501402,  # Belém (estuário, inclui-se por exposição portuária)
        1501709,  # Bragança
        1501956,  # Cachoeira do Piriá
        1502509,  # Chaves
        1502608,  # Colares
        1502905,  # Curuçá
        1504109,  # Magalhães Barata
        1504307,  # Maracanã
        1504406,  # Marapanim
        1506112,  # Quatipuru
        1506203,  # Salinópolis
        1506302,  # Salvaterra
        1506906,  # Santarém Novo
        1507102,  # São Caetano de Odivelas
        1507466,  # São João da Ponta
        1507904,  # Soure
        1508035,  # Tracuateua
        1508209,  # Vigia
        1508308,  # Viseu
    ]
)

_COSTEIROS_RN: frozenset[int] = frozenset(
    [
        2401107,  # Areia Branca
        2401404,  # Baía Formosa
        2401859,  # Caiçara do Norte
        2402204,  # Canguaretama
        2402600,  # Ceará-Mirim
        2403251,  # Parnamirim
        2403608,  # Extremoz
        2404101,  # Galinhos
        2404408,  # Grossos
        2404507,  # Guamaré
        2407203,  # Macau
        2407500,  # Maxaranguape
        2408102,  # Natal
        2408201,  # Nísia Floresta
        2408953,  # Rio do Fogo
        2409506,  # Pedra Grande
        2409902,  # Pendências
        2410256,  # Porto do Mangue
        2410405,  # Pureza
        2411056,  # Tibau
        2411601,  # São Bento do Norte
        2412005,  # São Gonçalo do Amarante
        2412559,  # São Miguel do Gostoso
        2413201,  # Senador Georgino Avelino
        2414209,  # Tibau do Sul
        2414407,  # Touros
        2415008,  # Vila Flor
        # Nota: Arês (nome com til especial) ainda não resolvido na API IBGE;
        # adicionar manualmente após verificação.
    ]
)

COSTEIROS: frozenset[int] = _COSTEIROS_MA | _COSTEIROS_AP | _COSTEIROS_PA | _COSTEIROS_RN
"""União dos 4 conjuntos: 80 municípios costeiros verificados."""


# ---------------------------------------------------------------------------
# Produtores atuais (ANP 2024 + hubs com blocos licitados)
# ---------------------------------------------------------------------------
PRODUTORES_ATUAIS: frozenset[int] = frozenset(
    [
        # RN — Bacia Potiguar (onshore + offshore ativos)
        2401107,  # Areia Branca
        2404507,  # Guamaré
        2407203,  # Macau
        2408003,  # Mossoró (interior, mas grande produtor onshore)
        2409902,  # Pendências
        2410256,  # Porto do Mangue
        # MA — hubs com blocos licitados na Margem Equatorial (Foz Amazonas)
        2111300,  # São Luís
        2101251,  # Bacabeira (refinaria)
        # AP — Oiapoque como hub de licitação Foz do Amazonas
        1600501,  # Oiapoque
        # PA — Belém como hub logístico para Pará-Maranhão
        1501402,  # Belém
    ]
)


@dataclass(frozen=True)
class ClassificationInputs:
    """Insumos para classificar municípios em zonas."""

    hdi_proxy_by_municipio: pd.Series
    """Series indexada por ``codigo_ibge`` com indicador de corte (PIB pc, IDHM...)."""

    hdi_median_by_uf: dict[str, float] | None = None
    """Medianas pré-computadas ``{uf_2_digitos: mediana}``; se vazio, calcula-se."""


def classify_municipio(
    codigo_ibge: int,
    inputs: ClassificationInputs,
) -> Zone:
    """Classifica um município em uma das 4 zonas."""
    is_coast = codigo_ibge in COSTEIROS
    is_producer = codigo_ibge in PRODUTORES_ATUAIS

    if is_coast and is_producer:
        return Zone.COSTA_PRODUTORA
    if is_coast:
        return Zone.COSTA_NAO_PRODUTORA

    # Produtor não-costeiro (ex.: Mossoró): tratamos como COSTA_PRODUTORA
    # pela lógica econômica (participa do mesmo cluster produtivo), mas a
    # spec pode ser refinada no futuro para uma 5ª zona "interior produtor".
    if is_producer:
        return Zone.COSTA_PRODUTORA

    uf = _uf_from_codigo(codigo_ibge)
    medians = inputs.hdi_median_by_uf or _compute_medians(inputs.hdi_proxy_by_municipio)
    valor = inputs.hdi_proxy_by_municipio.get(codigo_ibge)
    if pd.isna(valor) or valor is None:
        raise KeyError(f"hdi_proxy ausente para codigo_ibge={codigo_ibge}")
    if valor >= medians[uf]:
        return Zone.INTERIOR_MEDIO_ALTO
    return Zone.INTERIOR_BAIXO


def classify_dataframe(
    df: pd.DataFrame,
    codigo_col: str = "codigo_ibge",
    hdi_col: str = "pib_per_capita",
) -> pd.DataFrame:
    """Adiciona coluna ``zona`` a um DataFrame."""
    hdi_series = df.set_index(codigo_col)[hdi_col]
    inputs = ClassificationInputs(hdi_proxy_by_municipio=hdi_series)
    df = df.copy()
    df["zona"] = df[codigo_col].map(lambda c: classify_municipio(int(c), inputs).value)
    return df


# ---------------------------------------------------------------------------
# Agregação zonal — stocks e indicadores
# ---------------------------------------------------------------------------
def agregar_zonais(df_municipios: pd.DataFrame) -> pd.DataFrame:
    """Agrega DataFrame municipal em DataFrame zonal (UF x zona).

    Espera colunas: ``codigo_ibge``, ``uf``, ``zona``, ``populacao``,
    ``receita_corrente``, ``transferencias``, ``royalties_petroleo``,
    ``c_inst_proxy``, ``k_pub_inicial``, ``k_hum_inicial``, ``e_amb_inicial``.

    Stocks ponderados por população (K_pub, K_hum, C_inst); receita e
    população somadas.
    """

    def _wavg(sub: pd.DataFrame, col: str) -> float:
        w = sub["populacao"].astype(float)
        if w.sum() <= 0:
            return float("nan")
        return float((sub[col].astype(float) * w).sum() / w.sum())

    rows: list[dict[str, object]] = []
    for (uf, zona), sub in df_municipios.groupby(["uf", "zona"], dropna=False):
        rows.append(
            {
                "uf": uf,
                "zona": zona,
                "populacao": float(sub["populacao"].sum()),
                "receita_corrente": float(sub["receita_corrente"].sum()),
                "transferencias": float(sub["transferencias"].sum()),
                "royalties_petroleo": float(sub["royalties_petroleo"].sum()),
                "municipios": int(sub["codigo_ibge"].nunique()),
                "k_pub": _wavg(sub, "k_pub_inicial"),
                "k_hum": _wavg(sub, "k_hum_inicial"),
                "c_inst": _wavg(sub, "c_inst_proxy"),
                "e_amb": _wavg(sub, "e_amb_inicial"),
            }
        )
    out = pd.DataFrame.from_records(rows)
    if not out.empty:
        out["rent_per_capita"] = (out["royalties_petroleo"] / out["populacao"]).fillna(0.0)
    return out.sort_values(["uf", "zona"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gini zonal (§2.6)
# ---------------------------------------------------------------------------
def gini(values: pd.Series, weights: pd.Series | None = None) -> float:
    """Índice de Gini ponderado (0 = igualdade, 1 = concentração total)."""
    values = pd.Series(values).astype(float)
    if weights is None:
        weights = pd.Series(1.0, index=values.index)
    weights = pd.Series(weights).astype(float)
    order = values.argsort()
    v = values.iloc[order].to_numpy()
    w = weights.iloc[order].to_numpy()
    total_weight = w.sum()
    if total_weight <= 0:
        return 0.0
    mean = (v * w).sum() / total_weight
    if mean <= 0:
        return 0.0
    cumw = w.cumsum()
    return float(((2 * cumw - w) * v * w).sum() / (total_weight**2 * mean) - 1)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _uf_from_codigo(codigo_ibge: int) -> str:
    """Primeiros 2 dígitos do código IBGE-7 identificam a UF."""
    return f"{codigo_ibge:07d}"[:2]


def _compute_medians(series: pd.Series) -> dict[str, float]:
    """Mediana do indicador por UF (UF = prefixo 2 dígitos)."""
    df = series.reset_index()
    df.columns = ["codigo_ibge", "valor"]
    df["uf"] = df["codigo_ibge"].apply(lambda c: _uf_from_codigo(int(c)))
    return df.groupby("uf")["valor"].median().to_dict()
