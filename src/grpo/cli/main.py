"""Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from grpo.runner import run

app = typer.Typer(no_args_is_help=True, help="Small-scale GRPO training simulator.")
console = Console()


@app.command()
def info() -> None:
    console.print("grpo-small-reasoner: see `grpo bench --help`.")


@app.command()
def bench(
    out_dir: Path = typer.Option(Path("runs/latest")),
    n_steps: int = typer.Option(60),
    k: int = typer.Option(4),
    seed: int = typer.Option(17),
) -> None:
    res = run(out_dir, n_steps=n_steps, k=k, seed=seed)
    console.print_json(json.dumps({k: v for k, v in res.items() if k != "records"}, default=str))


if __name__ == "__main__":
    app()
