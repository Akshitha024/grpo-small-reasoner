"""End-to-end runner."""

from __future__ import annotations

import json
from pathlib import Path

from grpo.train.loop import train
from grpo.viz.charts import (
    headline_pareto,
    pass_at_curves,
    reward_distribution_hist,
    reward_over_steps,
    temperature_anneal,
)


def run(out_dir: Path, n_steps: int = 60, k: int = 4, seed: int = 17) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = Path("results/figures")
    records = train(n_steps=n_steps, k=k, seed=seed)
    reward_over_steps(records, figs / "reward_over_steps.png")
    pass_at_curves(records, figs / "pass_at_k.png")
    temperature_anneal(records, figs / "temperature.png")
    headline_pareto(records, figs / "pareto.png")
    reward_distribution_hist(records, figs / "reward_hist.png")

    summary: dict[str, object] = {
        "n_steps": len(records),
        "final_mean_reward": records[-1].mean_reward if records else 0.0,
        "final_pass_at_1": records[-1].pass_at_1 if records else 0.0,
        "final_pass_at_4": records[-1].pass_at_4 if records else 0.0,
        "records": [r.model_dump() for r in records],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
