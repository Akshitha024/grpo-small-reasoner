"""Tests for the simulated policy."""

from __future__ import annotations

from grpo.policy.sim import SimPolicy


def test_p_correct_grows_with_step() -> None:
    p = SimPolicy(seed=1)
    p0 = p.p_correct("t-01")
    p.step = 30
    p1 = p.p_correct("t-01")
    assert p1 > p0


def test_temperature_decays() -> None:
    p = SimPolicy()
    t0 = p.temperature()
    p.step = 50
    t1 = p.temperature()
    assert t1 < t0


def test_per_task_ceiling_caches() -> None:
    p = SimPolicy()
    c = p.per_task_ceiling("t-01")
    assert p.per_task_ceiling("t-01") == c
