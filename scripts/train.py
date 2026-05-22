"""CLI de treino BRO-MARL.

Uso:
    uv run python -m scripts.train --episodes 200 --uf MA --scenario referencia
"""

from __future__ import annotations

import logging

import typer

from agents.bro_marl import BROMARLSystem
from agents.definitions import state_dim_global
from agents.trainer import Trainer, TrainerConfig
from core.environment import MarginPlayEnv

app = typer.Typer(help="Treino BRO-MARL no Margin Play.")


@app.command()
def train(
    episodes: int = typer.Option(200, help="Número de episódios."),
    uf: str = typer.Option("MA", help="UF-alvo (MA, AP, PA, RN)."),
    scenario: str = typer.Option("referencia", help="Cenário (pessimista, referencia, otimista)."),
    batch_size: int = typer.Option(256, help="Batch size."),
    warmup: int = typer.Option(5, help="Episódios de warmup antes do primeiro update."),
    log_every: int = typer.Option(25, help="Log a cada N episódios."),
    seed: int = typer.Option(42, help="Random seed."),
    buffer_capacity: int = typer.Option(100_000, help="Capacidade do replay buffer."),
) -> None:
    """Roda treino MADDPG e imprime métricas a cada bloco."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    env = MarginPlayEnv(uf=uf, scenario=scenario, rng_seed=seed)
    system = BROMARLSystem.build(state_dim=state_dim_global(), buffer_capacity=buffer_capacity)
    config = TrainerConfig(
        n_episodes=episodes,
        batch_size=batch_size,
        warmup_episodes=warmup,
        log_every_episodes=log_every,
        seed=seed,
    )
    trainer = Trainer(env, system, config)

    typer.echo(
        f"=== Margin Play training: UF={uf} cenário={scenario} ep={episodes} batch={batch_size} ==="
    )
    stats = trainer.run()

    typer.echo("\n=== Resumo final ===")
    last = stats[-1]
    for ag, ret in last.returns.items():
        typer.echo(f"  {ag:14}: return final = {ret:+.3f}")
    typer.echo(
        f"  W final={last.final_W:.3f}  E_amb final={last.final_E_amb:.3f}  "
        f"R final={last.final_R:.2f} Gbbl"
    )


if __name__ == "__main__":
    app()
