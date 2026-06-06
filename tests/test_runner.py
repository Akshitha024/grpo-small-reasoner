"""End-to-end runner."""

from __future__ import annotations

from pathlib import Path

from grpo.runner import run


def test_runner_smoke(tmp_path: Path) -> None:
    s = run(tmp_path / "out", n_steps=20, k=4, seed=1)
    assert s["n_steps"] == 20
    assert (tmp_path / "out" / "summary.json").exists()
