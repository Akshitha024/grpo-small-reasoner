"""Training-loop tests."""

from __future__ import annotations

from grpo.train.loop import train


def test_train_records_each_step() -> None:
    rec = train(n_steps=10)
    assert len(rec) == 10


def test_reward_increases_over_steps() -> None:
    rec = train(n_steps=40)
    early = sum(r.mean_reward for r in rec[:5]) / 5
    late = sum(r.mean_reward for r in rec[-5:]) / 5
    assert late > early
