"""Simulated policy whose competence improves over training steps.

We model the policy as a per-task success probability that starts low and
converges to a per-task ceiling. Roll-outs are Bernoulli draws against this
probability. The temperature shrinks with training (annealing).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class SimPolicy:
    seed: int = 17
    base_competence: float = 0.10
    ceiling: float = 0.92
    learning_rate: float = 0.06
    temp_initial: float = 1.0
    temp_floor: float = 0.2
    step: int = 0
    _per_task_ceiling: dict[str, float] = field(default_factory=dict)

    def temperature(self) -> float:
        decay = max(0.0, 1.0 - 0.01 * self.step)
        return max(self.temp_floor, self.temp_initial * decay)

    def per_task_ceiling(self, tid: str) -> float:
        # Easy tasks (t-01..t-04) get a higher ceiling; harder tasks lower.
        if tid not in self._per_task_ceiling:
            offset = 0.0 if tid in ("t-01", "t-02", "t-03", "t-04") else -0.15
            self._per_task_ceiling[tid] = self.ceiling + offset
        return self._per_task_ceiling[tid]

    def p_correct(self, tid: str) -> float:
        # Logistic-like growth toward the per-task ceiling.
        ceil = self.per_task_ceiling(tid)
        progress = 1.0 - (1.0 - self.learning_rate) ** self.step
        return self.base_competence + (ceil - self.base_competence) * progress

    def rollout(self, tid: str, rollout_idx: int) -> tuple[str, float]:
        """Returns (output, reward)."""
        rng = random.Random(hash((self.seed, self.step, tid, rollout_idx)) & 0xFFFFFFFF)
        if rng.random() < self.p_correct(tid):
            return ("CORRECT", 1.0)
        return ("WRONG", 0.0)

    def update(self) -> None:
        self.step += 1
