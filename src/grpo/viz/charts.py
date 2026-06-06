"""Five chart families for grpo-small-reasoner."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from grpo.types import StepRecord


def _save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def reward_over_steps(records: list[StepRecord], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [r.step for r in records]
    ax.plot(steps, [r.mean_reward for r in records], label="mean reward", color="#3b6fa1")
    ax.set_xlabel("step")
    ax.set_ylabel("mean reward")
    ax.set_title("Mean reward vs training step")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    return _save(fig, out)


def pass_at_curves(records: list[StepRecord], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [r.step for r in records]
    ax.plot(steps, [r.pass_at_1 for r in records], label="pass@1", color="#c25a4f")
    ax.plot(steps, [r.pass_at_4 for r in records], label="pass@4", color="#5b8d4a")
    ax.set_xlabel("step")
    ax.set_ylabel("pass rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("pass@k over training")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out)


def temperature_anneal(records: list[StepRecord], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [r.step for r in records]
    ax.plot(steps, [r.policy_temp for r in records], color="#5b8d4a")
    ax.set_xlabel("step")
    ax.set_ylabel("policy temperature")
    ax.set_title("Temperature annealing")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def headline_pareto(records: list[StepRecord], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r.policy_temp for r in records]
    ys = [r.mean_reward for r in records]
    sc = ax.scatter(xs, ys, c=[r.step for r in records], cmap="viridis", s=20)
    fig.colorbar(sc, ax=ax, label="step")
    ax.set_xlabel("policy temperature")
    ax.set_ylabel("mean reward")
    ax.set_title("Temperature vs reward (colored by step)")
    return _save(fig, out)


def reward_distribution_hist(records: list[StepRecord], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    rewards = np.array([r.mean_reward for r in records])
    ax.hist(rewards, bins=20, color="#3b6fa1", edgecolor="white")
    ax.set_xlabel("mean reward (per training step)")
    ax.set_ylabel("# steps")
    ax.set_title("Distribution of mean reward across the training run")
    return _save(fig, out)
