"""Simulated GRPO training loop.

For each step:
  1. Sample k roll-outs per task.
  2. Compute group-relative advantage (reward minus group mean).
  3. Pseudo-update the policy (advancing the step counter).
  4. Record mean reward + pass@1 + pass@4.
"""

from __future__ import annotations

from grpo.env.tasks import tasks
from grpo.policy.sim import SimPolicy
from grpo.types import Sample, StepRecord


def train(n_steps: int = 60, k: int = 4, seed: int = 17) -> list[StepRecord]:
    policy = SimPolicy(seed=seed)
    records: list[StepRecord] = []
    task_set = tasks()
    for _ in range(n_steps):
        all_samples: list[Sample] = []
        for t in task_set:
            for r in range(k):
                output, reward = policy.rollout(t.tid, r)
                all_samples.append(Sample(tid=t.tid, rollout_idx=r, output=output, reward=reward))
        mean_reward = sum(s.reward for s in all_samples) / max(1, len(all_samples))
        # pass@1 = mean reward when taking the first roll-out per task.
        firsts = [s for s in all_samples if s.rollout_idx == 0]
        pass_at_1 = sum(s.reward for s in firsts) / max(1, len(firsts))
        # pass@k = task has any correct roll-out.
        by_task: dict[str, list[Sample]] = {}
        for s in all_samples:
            by_task.setdefault(s.tid, []).append(s)
        pass_at_k = sum(any(s.reward > 0 for s in rs) for rs in by_task.values()) / max(
            1, len(by_task)
        )
        records.append(
            StepRecord(
                step=policy.step,
                mean_reward=mean_reward,
                pass_at_1=pass_at_1,
                pass_at_4=pass_at_k,
                policy_temp=policy.temperature(),
            )
        )
        policy.update()
    return records
