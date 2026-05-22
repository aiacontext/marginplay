"""Sweep multi-cenário com persistência de métricas, checkpoints e metadados.

Roda treino BRO-MARL para N cenários sequencialmente e grava (em ``out_dir``):

- ``{cenario}_episodes.parquet`` — uma linha por episódio com returns,
  losses (actor/critic), métricas distributional do critic (q_target_mean,
  q_target_max, reward_mean_batch) e finais W/E_amb/R.
- ``{cenario}_actor_<agent>.npz`` — pesos finais de cada actor.
- ``{cenario}_critic_<agent>.npz`` — pesos finais de cada critic distributional.
- ``checkpoints/{cenario}/ep{N}/{actor|critic}_<agent>.npz`` — snapshots
  periódicos a cada ``checkpoint_every`` episódios.
- ``sweep_summary.parquet`` — agregado por cenário (último decil + eval).
- ``sweep_metadata.json`` — commit_sha, seed, hyperparams, timing.
- ``sweep_<timestamp>.log`` — captura completa do log do sweep.

Uso:
    uv run python -m scripts.sweep --episodes 10000 \\
        --scenarios pessimista,referencia,otimista --checkpoint-every 1000
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from agents.bro_marl import (
    ACTOR_LR,
    CRITIC_LR,
    GAMMA_DEFAULT,
    HUBER_KAPPA,
    N_QUANTILES,
    TAU_DEFAULT,
    BROMARLSystem,
)
from agents.definitions import state_dim_global
from agents.trainer import EpisodeStats, Trainer, TrainerConfig
from core.environment import AGENTS, MarginPlayEnv

app = typer.Typer(help="Sweep multi-cenário Margin Play.")
console = Console()
logger = logging.getLogger(__name__)


def _save_episodes_parquet(stats: list[EpisodeStats], path: Path, scenario: str) -> None:
    """Persiste métricas por episódio. Inclui campos BRO-MARL distributional."""
    rows = []
    for s in stats:
        row = {
            "scenario": scenario,
            "episode": s.episode,
            "final_W": s.final_W,
            "final_E_amb": s.final_E_amb,
            "final_R": s.final_R,
        }
        for ag in AGENTS:
            row[f"return_{ag}"] = s.returns.get(ag, float("nan"))
            row[f"critic_{ag}"] = s.losses_critic.get(ag, float("nan"))
            row[f"actor_{ag}"] = s.losses_actor.get(ag, float("nan"))
            row[f"q_target_mean_{ag}"] = s.q_target_mean.get(ag, float("nan"))
            row[f"q_target_max_{ag}"] = s.q_target_max.get(ag, float("nan"))
            row[f"reward_mean_{ag}"] = s.reward_mean_batch.get(ag, float("nan"))
        rows.append(row)
    df = pd.DataFrame.from_records(rows)
    df.to_parquet(path, compression="zstd")


def _flatten_params(params: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten_params(item, f"{key}.{i}"))
                else:
                    out[f"{key}.{i}"] = item
        else:
            out[key] = v
    return out


def _save_network(net: object, path: Path) -> None:
    """Achata parâmetros de uma nn.Module e salva como .npz."""
    params_flat = _flatten_params(net.parameters())  # type: ignore[attr-defined]
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path), **params_flat)


def _save_actors_and_critics(system: BROMARLSystem, dest_dir: Path, scenario: str) -> None:
    """Salva pesos finais de actor e critic de cada agente."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ag, agent in system.agents.items():
        _save_network(agent.actor, dest_dir / f"{scenario}_actor_{ag}.npz")
        _save_network(agent.critic, dest_dir / f"{scenario}_critic_{ag}.npz")


def _save_checkpoint(system: BROMARLSystem, ckpt_dir: Path, scenario: str, ep: int) -> None:
    """Snapshot periódico de actor + critic de cada agente."""
    target = ckpt_dir / scenario / f"ep{ep:06d}"
    target.mkdir(parents=True, exist_ok=True)
    for ag, agent in system.agents.items():
        _save_network(agent.actor, target / f"actor_{ag}.npz")
        _save_network(agent.critic, target / f"critic_{ag}.npz")


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _setup_file_logging(log_path: Path) -> logging.FileHandler:
    """Adiciona FileHandler ao root logger; retorna o handler para cleanup."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _evaluate_deterministic(
    env: MarginPlayEnv,
    system: BROMARLSystem,
    n_episodes: int = 5,
) -> dict[str, float]:
    """Roda episódios sem exploração e devolve métricas médias finais."""
    metrics_w: list[float] = []
    metrics_e: list[float] = []
    metrics_r: list[float] = []
    metrics_ret: dict[str, list[float]] = {ag: [] for ag in AGENTS}

    for _ in range(n_episodes):
        obs = env.reset()
        cum: dict[str, float] = {ag: 0.0 for ag in AGENTS}
        last = None
        for _ in range(env.world.horizon_steps):  # type: ignore[union-attr]
            actions = system.act(obs, explore=False)
            result = env.step(actions)
            for ag, r in result.rewards.items():
                cum[ag] += r
            obs = result.observations
            last = result.info
            if result.done:
                break
        metrics_w.append(last.W if last else 0.0)
        metrics_e.append(env.world.state.E_amb)  # type: ignore[union-attr]
        metrics_r.append(env.world.state.R)  # type: ignore[union-attr]
        for ag in AGENTS:
            metrics_ret[ag].append(cum[ag])

    return {
        "eval_W_mean": float(np.mean(metrics_w)),
        "eval_E_amb_mean": float(np.mean(metrics_e)),
        "eval_R_mean": float(np.mean(metrics_r)),
        **{f"eval_return_{ag}": float(np.mean(metrics_ret[ag])) for ag in AGENTS},
    }


def _train_one_scenario(
    scenario: str,
    uf: str,
    episodes: int,
    batch_size: int,
    warmup: int,
    log_every: int,
    seed: int,
    buffer_capacity: int,
    out_dir: Path,
    checkpoint_every: int,
) -> dict[str, object]:
    """Treina um cenário, grava artefatos e devolve resumo."""
    console.print(f"\n[bold cyan]>>> Iniciando cenário: {scenario}[/] (UF={uf}, ep={episodes})")
    logger.info("Iniciando cenário=%s uf=%s ep=%d seed=%d", scenario, uf, episodes, seed)
    t_start = time.time()

    env = MarginPlayEnv(uf=uf, scenario=scenario, rng_seed=seed)
    system = BROMARLSystem.build(state_dim=state_dim_global(), buffer_capacity=buffer_capacity)
    config = TrainerConfig(
        n_episodes=episodes,
        batch_size=batch_size,
        warmup_episodes=warmup,
        log_every_episodes=log_every,
        seed=seed,
    )

    ckpt_dir = out_dir / "checkpoints"

    def _on_episode_end(ep_idx: int, _stats: EpisodeStats, sys: BROMARLSystem) -> None:
        # Checkpoint a cada ``checkpoint_every`` episódios (1-indexed).
        if checkpoint_every > 0 and (ep_idx + 1) % checkpoint_every == 0:
            _save_checkpoint(sys, ckpt_dir, scenario, ep_idx + 1)
            logger.info("checkpoint salvo: %s ep=%d", scenario, ep_idx + 1)

    trainer = Trainer(env, system, config, on_episode_end=_on_episode_end)
    stats = trainer.run()
    elapsed = time.time() - t_start

    # Persistir métricas e estado terminal (actors + critics)
    parquet_path = out_dir / f"{scenario}_episodes.parquet"
    _save_episodes_parquet(stats, parquet_path, scenario)
    _save_actors_and_critics(system, out_dir, scenario)

    # Evaluation determinístico
    console.print(f"  Evaluating deterministic policy ({scenario})...")
    eval_metrics = _evaluate_deterministic(env, system, n_episodes=5)

    # Resumo último decil
    last_decile = stats[max(1, len(stats) // 10) :]
    avg_returns = {ag: float(np.mean([s.returns[ag] for s in last_decile])) for ag in AGENTS}
    avg_w = float(np.mean([s.final_W for s in last_decile]))
    avg_e = float(np.mean([s.final_E_amb for s in last_decile]))

    console.print(
        f"[green]<<< {scenario} concluído em {elapsed / 60:.1f} min[/] "
        f"({elapsed / episodes * 1000:.0f} ms/ep)"
    )
    logger.info(
        "Cenário concluído: %s elapsed_min=%.2f ms_per_ep=%.1f",
        scenario,
        elapsed / 60,
        elapsed / episodes * 1000,
    )

    return {
        "scenario": scenario,
        "episodes": episodes,
        "elapsed_s": elapsed,
        "ms_per_episode": elapsed / episodes * 1000,
        "train_W_mean_lastdecile": avg_w,
        "train_E_amb_mean_lastdecile": avg_e,
        **{f"train_return_{ag}_lastdecile": avg_returns[ag] for ag in AGENTS},
        **eval_metrics,
    }


@app.command()
def sweep(
    episodes: int = typer.Option(1000, help="Episódios por cenário."),
    scenarios: str = typer.Option(
        "pessimista,referencia,otimista", help="Cenários separados por vírgula."
    ),
    uf: str = typer.Option("MA", help="UF-alvo."),
    batch_size: int = typer.Option(256, help="Batch size."),
    warmup: int = typer.Option(5, help="Episódios de warmup."),
    log_every: int = typer.Option(100, help="Log a cada N episódios."),
    seed: int = typer.Option(42, help="Random seed base."),
    buffer_capacity: int = typer.Option(100_000, help="Capacidade replay."),
    checkpoint_every: int = typer.Option(
        1000, help="Snapshot de actors+critics a cada N eps (0 desativa)."
    ),
    out_dir: Path = typer.Option(  # noqa: B008 -- padrão idiomático typer
        Path("models"), help="Diretório de artefatos."
    ),
) -> None:
    """Roda treino BRO-MARL para múltiplos cenários e gera comparativo."""
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = out_dir / f"sweep_{timestamp}.log"

    # Configurar logging: stream (console) + file capture
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    file_handler = _setup_file_logging(log_path)
    logger.info("Sweep iniciado timestamp=%s log=%s", timestamp, log_path)

    scenario_list = [s.strip() for s in scenarios.split(",") if s.strip()]
    summaries: list[dict[str, object]] = []
    t_total = time.time()

    try:
        for i, sc in enumerate(scenario_list):
            summary = _train_one_scenario(
                scenario=sc,
                uf=uf,
                episodes=episodes,
                batch_size=batch_size,
                warmup=warmup,
                log_every=log_every,
                seed=seed + i,
                buffer_capacity=buffer_capacity,
                out_dir=out_dir,
                checkpoint_every=checkpoint_every,
            )
            summaries.append(summary)

        df_sum = pd.DataFrame.from_records(summaries)
        summary_path = out_dir / "sweep_summary.parquet"
        df_sum.to_parquet(summary_path, compression="zstd")

        total_s = time.time() - t_total
        total_min = total_s / 60

        # Metadata da rodada
        metadata = {
            "timestamp": timestamp,
            "git_commit_sha": _git_commit_sha(),
            "uf": uf,
            "scenarios": scenario_list,
            "episodes_per_scenario": episodes,
            "seed_base": seed,
            "batch_size": batch_size,
            "warmup_episodes": warmup,
            "buffer_capacity": buffer_capacity,
            "checkpoint_every": checkpoint_every,
            "elapsed_s_total": total_s,
            "log_path": str(log_path.relative_to(out_dir.parent))
            if log_path.is_relative_to(out_dir.parent)
            else str(log_path),
            "summary_path": str(summary_path.name),
            "hyperparams": {
                "gamma": GAMMA_DEFAULT,
                "tau": TAU_DEFAULT,
                "actor_lr": ACTOR_LR,
                "critic_lr": CRITIC_LR,
                "n_quantiles": N_QUANTILES,
                "huber_kappa": HUBER_KAPPA,
            },
        }
        meta_path = out_dir / "sweep_metadata.json"
        with meta_path.open("w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Metadata salvo: %s", meta_path)

        console.print(f"\n[bold]Sweep total: {total_min:.1f} min[/]")
        console.print(f"Resumo:   [cyan]{summary_path}[/]")
        console.print(f"Metadata: [cyan]{meta_path}[/]")
        console.print(f"Log:      [cyan]{log_path}[/]")

        # Tabela comparativa
        table = Table(title="Comparativo de cenários (último decil + eval determinístico)")
        table.add_column("Cenário", style="cyan")
        table.add_column("W (train)")
        table.add_column("W (eval)")
        table.add_column("E_amb (eval)")
        table.add_column("R (eval)")
        table.add_column("Return Gov_Fed (train)")
        table.add_column("Return ANP (train)")
        table.add_column("Return IBAMA (train)")
        table.add_column("Return Com (train)")
        for s in summaries:
            table.add_row(
                str(s["scenario"]),
                f"{s['train_W_mean_lastdecile']:.3f}",
                f"{s['eval_W_mean']:.3f}",
                f"{s['eval_E_amb_mean']:.3f}",
                f"{s['eval_R_mean']:.2f}",
                f"{s['train_return_gov_federal_lastdecile']:+.2f}",
                f"{s['train_return_anp_lastdecile']:+.2f}",
                f"{s['train_return_ibama_lastdecile']:+.2f}",
                f"{s['train_return_comunidade_lastdecile']:+.2f}",
            )
        console.print(table)
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    app()
