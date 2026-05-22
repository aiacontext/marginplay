# Margin Play

Multi-agent reinforcement learning simulation of oil exploration in the Brazilian Equatorial Margin. Six trained agents (operator, ANP, IBAMA, federal government, state government, community) interact in a coupled socio-economic-environmental world over monthly steps.

**Stack:** Python 3.12 + [MLX](https://github.com/ml-explore/mlx) (Apple Silicon) + MADDPG.

## Repository layout

```
core/      Simulation engine (stocks, transitions, scenarios, geography)
agents/    MADDPG in MLX (Actor/Critic, replay buffer, trainer, noise)
scripts/   CLI (train, sweep, analyze, parity fixtures)
tests/     Unit tests
```

## Trained checkpoints

Pretrained weights live on Hugging Face: **[`aiacontext/marginplay`](https://huggingface.co/aiacontext/marginplay)**.

The published sweep `sweep_v6_6sc_10k` contains 6 scenarios × {6 actors, 6 critics} of `.npz` files plus per-scenario episode logs in Parquet.

```bash
# pull weights into ./models/
uv run --group hf hf download aiacontext/marginplay --local-dir models/
```

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync                  # core deps
uv sync --all-groups     # core + dev + hf
```

## Commands

```bash
uv run pytest                                              # tests
uv run ruff check .                                        # lint
uv run ruff format .                                       # format
uv run python -m scripts.train --uf MA --scenario referencia --episodes 200
uv run python -m scripts.sweep --episodes 10000            # multi-scenario sweep
uv run python -m scripts.analyze_sweep                     # diagnostics
```

## Inference (deterministic replay)

The Actor network is a small MLP; weights load with `numpy.load` from the `.npz` files. With `explore=False` the policy is deterministic — given the same scenario seed and intervention log, the trajectory is reproducible.

## Conventions

- Python 3.12 (pinned in `.python-version`).
- Package manager: `uv` (not pip/poetry/conda).
- Lint/format: `ruff`. Type hints required on new code; `mypy --strict` is optional.
- Code in English; docstrings/comments in Portuguese.
- Data join keys: IBGE-7 (municipality), ANP field/block code, monthly timestamp.

## Citation

If you use Margin Play or its pretrained weights in academic work, please cite:

```bibtex
@unpublished{leitaofilho2026marginplay,
  title  = {Margin Play: A Multi-Agent System for Public Policy Analysis
            in the Brazilian Equatorial Margin},
  author = {Leit{\~a}o Filho, Antonio de Sousa and
            Lima, Fabr{\'\i}cio Saul and
            Santos, Selby Mykael Lima dos and
            Sousa, Rejani Bandeira Vieira and
            Jesus, Lu{\'\i}s Jorge Mesquita de and
            Silva, Dennys Correia da and
            Barros Filho, Allan Kardec Duailibe},
  year   = {2026},
  note   = {Manuscript in preparation},
}
```

Plain text:

> Leitão Filho, A. S., Lima, F. S., Santos, S. M. L. dos, Sousa, R. B. V., Jesus, L. J. M. de, Silva, D. C. da, & Barros Filho, A. K. D. (2026). *Margin Play: A Multi-Agent System for Public Policy Analysis in the Brazilian Equatorial Margin*. Manuscript in preparation.

**Corresponding author:** Antonio de Sousa Leitão Filho — [antonio@aiacontext.com](mailto:antonio@aiacontext.com) — ORCID [0009-0002-1705-3611](https://orcid.org/0009-0002-1705-3611).

## License

Released under the [MIT License](LICENSE).
