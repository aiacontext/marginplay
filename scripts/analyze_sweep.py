"""Análise dos parquets gerados pelo sweep.

Lê ``models/{cenario}_episodes.parquet`` para cada cenário, computa:
- Curvas de aprendizado (rolling mean de returns por agente, W, E_amb).
- Convergência (variância dos returns nos últimos 1000 eps).
- Tabela final comparativa.

Uso:
    uv run python -m scripts.analyze_sweep
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from core.environment import AGENTS

app = typer.Typer(help="Análise comparativa do sweep multi-cenário.")
console = Console()


@app.command()
def analyze(
    models_dir: Path = typer.Option(  # noqa: B008
        Path("models"), help="Diretório com parquets do sweep."
    ),
    last_n: int = typer.Option(1000, help="Últimos N episódios para estatísticas finais."),
) -> None:
    """Carrega parquets do sweep e gera comparativo enriquecido."""
    summary_path = models_dir / "sweep_summary.parquet"
    if not summary_path.exists():
        console.print(f"[red]sweep_summary.parquet não encontrado em {models_dir}[/]")
        raise typer.Exit(1)

    summary = pd.read_parquet(summary_path)
    console.print("\n[bold]Resumo do sweep (já presente em sweep_summary.parquet):[/]")
    console.print(summary.to_string(index=False))

    # Carrega episode-level
    by_scenario: dict[str, pd.DataFrame] = {}
    for sc in summary["scenario"].tolist():
        path = models_dir / f"{sc}_episodes.parquet"
        if path.exists():
            by_scenario[sc] = pd.read_parquet(path)

    # Análise de convergência (variância dos returns nos últimos N eps)
    console.print(f"\n[bold]Variância de returns nos últimos {last_n} episódios:[/]")
    table = Table(title=f"Estabilidade da política (std return em últimos {last_n} eps)")
    table.add_column("Cenário", style="cyan")
    for ag in AGENTS:
        table.add_column(ag.split("_")[0])
    for sc, df in by_scenario.items():
        tail = df.tail(last_n)
        row = [sc]
        for ag in AGENTS:
            std = float(tail[f"return_{ag}"].std())
            row.append(f"{std:.3f}")
        table.add_row(*row)
    console.print(table)

    # Trajetória de W ao longo do treino (5 buckets)
    console.print("\n[bold]Trajetória de W (média por bucket de eps/5):[/]")
    table_w = Table(title="W médio por bucket de episódios")
    table_w.add_column("Cenário", style="cyan")
    n_buckets = 5
    for b in range(n_buckets):
        table_w.add_column(f"q{b + 1}/{n_buckets}")
    for sc, df in by_scenario.items():
        chunks = np.array_split(df["final_W"].to_numpy(), n_buckets)
        row = [sc] + [f"{np.nanmean(c):.3f}" for c in chunks]
        table_w.add_row(*row)
    console.print(table_w)

    # Trajetória de E_amb
    console.print("\n[bold]Trajetória de E_amb (média por bucket):[/]")
    table_e = Table(title="E_amb médio por bucket de episódios")
    table_e.add_column("Cenário", style="cyan")
    for b in range(n_buckets):
        table_e.add_column(f"q{b + 1}/{n_buckets}")
    for sc, df in by_scenario.items():
        chunks = np.array_split(df["final_E_amb"].to_numpy(), n_buckets)
        row = [sc] + [f"{np.nanmean(c):.3f}" for c in chunks]
        table_e.add_row(*row)
    console.print(table_e)

    # Trajetória de returns dos agentes mais reveladores (split regulador → ANP+IBAMA na v2)
    for ag in ("gov_federal", "anp", "ibama", "comunidade"):
        console.print(f"\n[bold]Trajetória de return_{ag} (média por bucket):[/]")
        table_ret = Table(title=f"return_{ag} médio por bucket")
        table_ret.add_column("Cenário", style="cyan")
        for b in range(n_buckets):
            table_ret.add_column(f"q{b + 1}/{n_buckets}")
        for sc, df in by_scenario.items():
            chunks = np.array_split(df[f"return_{ag}"].to_numpy(), n_buckets)
            row = [sc] + [f"{np.nanmean(c):+.2f}" for c in chunks]
            table_ret.add_row(*row)
        console.print(table_ret)


if __name__ == "__main__":
    app()
