"""Gera fixtures de paridade MLX → NumPy.

Para cada (cenário, agente) carrega o ``.npz`` no Actor MLX, gera N
observações aleatórias seedadas e salva os outputs do MLX como referência.
O ``parity_test.py`` em ``web/`` depois carrega o mesmo ``.npz`` no
``NumpyActor`` e compara contra essas referências — sem precisar de MLX
no runtime de inferência.

Uso (a partir da raiz do parent ``Margin_Play/``):

    uv run python -m scripts.gen_parity_fixtures \\
        --output web/api/_engine/fixtures/parity.npz \\
        --weights-dir models/sweep_v6_6sc_10k \\
        --scenarios referencia,otimista,pessimista,choque_brent,ma_prospero,sem_lei12858 \\
        --n-samples 256 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from agents.definitions import SPECS
from agents.networks import Actor
from core.environment import AGENTS


def gen_for_actor(
    actor_npz: Path, agent_id: str, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Carrega Actor MLX, sorteia obs em [-2, 2], roda forward, devolve (obs, out)."""
    spec = SPECS[agent_id]
    actor = Actor(spec.obs_dim, spec.act_dim, output_activation=spec.output_activation)
    actor.load_weights(str(actor_npz))

    obs = rng.uniform(-2.0, 2.0, size=(n_samples, spec.obs_dim)).astype(np.float32)
    out_mx = actor(mx.array(obs))
    out_np = np.asarray(out_mx).astype(np.float32)
    return obs, out_np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="referencia,otimista,pessimista,choque_brent,ma_prospero,sem_lei12858",
    )
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]
    rng = np.random.default_rng(args.seed)

    fixtures: dict[str, np.ndarray] = {}
    pairs_count = 0
    for scenario in scenarios:
        for agent_id in AGENTS:
            actor_npz = args.weights_dir / f"{scenario}_actor_{agent_id}.npz"
            if not actor_npz.exists():
                print(f"  SKIP missing: {actor_npz}")
                continue
            obs, out = gen_for_actor(actor_npz, agent_id, args.n_samples, rng)
            fixtures[f"{scenario}/{agent_id}/obs"] = obs
            fixtures[f"{scenario}/{agent_id}/out"] = out
            pairs_count += 1
            print(f"  {scenario}/{agent_id}: obs{obs.shape} out{out.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), **fixtures)
    print(f"\nWrote {pairs_count} (obs, out) pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
