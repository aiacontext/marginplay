"""Gera fixtures de paridade end-to-end via episódios MLX.

Diferente de ``gen_parity_fixtures.py`` (obs aleatórias uniformes), este
script roda o ambiente real (``MarginPlayEnv``) por 15 steps com os
actors MLX em modo determinístico (``explore=False`` equivalente),
capturando a trajetória completa: observações reais e ações tanto **raw**
(saída do actor) quanto **rescaled** (após ``AgentSpec.rescale_action``).

A vantagem é cobrir a distribuição operacional real — onde os agentes
de fato vão atuar em produção — em vez de uma distribuição uniforme que
pode passar perto de regimes degenerados (ex.: obs de baixa variância
amplificadas por LayerNorm).

Uso (a partir da raiz do parent):

    uv run python -m scripts.gen_parity_episode_fixtures \\
        --output web/api/_engine/fixtures/episode_parity.npz \\
        --weights-dir models/sweep_v6_6sc_10k \\
        --uf MA --seed 42

Saída: ``.npz`` com chaves ``{scenario}/{kind}/{agent}`` onde ``kind`` ∈
``{obs, raw, rescaled}``, cada array shape ``(n_steps, dim_correspondente)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from agents.definitions import SPECS
from agents.networks import Actor
from core.environment import AGENTS, MarginPlayEnv

DEFAULT_SCENARIOS = (
    "referencia",
    "otimista",
    "pessimista",
    "choque_brent",
    "ma_prospero",
    "sem_lei12858",
)


def _load_mlx_actor(path: Path, agent_id: str) -> Actor:
    spec = SPECS[agent_id]
    actor = Actor(spec.obs_dim, spec.act_dim, output_activation=spec.output_activation)
    actor.load_weights(str(path))
    return actor


def gen_episode(
    scenario: str, uf: str, weights_dir: Path, seed: int
) -> dict[str, np.ndarray]:
    """Roda 1 episódio com actors MLX em modo determinístico, captura
    (obs, raw, rescaled) por agente em cada step. Retorna dict pronto pra
    serializar — chaves ``{kind}/{agent}`` (sem prefixo de cenário ainda)."""
    actors = {
        ag: _load_mlx_actor(weights_dir / f"{scenario}_actor_{ag}.npz", ag) for ag in AGENTS
    }
    env = MarginPlayEnv(uf=uf, scenario=scenario, rng_seed=seed)
    obs = env.reset()
    horizon = env.world.horizon_steps  # type: ignore[union-attr]

    obs_log: dict[str, list[np.ndarray]] = {ag: [] for ag in AGENTS}
    raw_log: dict[str, list[np.ndarray]] = {ag: [] for ag in AGENTS}
    rescaled_log: dict[str, list[np.ndarray]] = {ag: [] for ag in AGENTS}

    for _ in range(horizon):
        actions: dict[str, np.ndarray] = {}
        for ag in AGENTS:
            spec = SPECS[ag]
            obs_arr = obs[ag].astype(np.float32)
            raw_mx = actors[ag](mx.array(obs_arr)[None, :])[0]
            raw_np = np.asarray(raw_mx).astype(np.float32)
            rescaled = spec.rescale_action(raw_np).astype(np.float32)

            obs_log[ag].append(obs_arr)
            raw_log[ag].append(raw_np)
            rescaled_log[ag].append(rescaled)
            actions[ag] = rescaled

        result = env.step(actions)
        obs = result.observations
        if result.done:
            break

    out: dict[str, np.ndarray] = {}
    for ag in AGENTS:
        out[f"obs/{ag}"] = np.stack(obs_log[ag], axis=0)
        out[f"raw/{ag}"] = np.stack(raw_log[ag], axis=0)
        out[f"rescaled/{ag}"] = np.stack(rescaled_log[ag], axis=0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--scenarios", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--uf", type=str, default="MA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]

    fixtures: dict[str, np.ndarray] = {}
    for scenario in scenarios:
        if not all((args.weights_dir / f"{scenario}_actor_{ag}.npz").exists() for ag in AGENTS):
            print(f"  SKIP {scenario}: pesos incompletos em {args.weights_dir}")
            continue

        episode = gen_episode(scenario, args.uf, args.weights_dir, args.seed)
        n_steps = next(iter(episode.values())).shape[0]
        for k, v in episode.items():
            fixtures[f"{scenario}/{k}"] = v
        print(f"  {scenario}: {n_steps} steps × 6 agentes (uf={args.uf}, seed={args.seed})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), **fixtures)
    n_pairs = len(fixtures) // 3  # obs + raw + rescaled per (scenario, agent)
    print(f"\nWrote {n_pairs} (scenario × agent) episodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
