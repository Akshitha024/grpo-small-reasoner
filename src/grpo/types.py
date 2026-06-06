"""Types."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Task(BaseModel):
    tid: str
    question: str
    answer: str


class Sample(BaseModel):
    tid: str
    rollout_idx: int
    output: str
    reward: float = Field(ge=0, le=1)


class StepRecord(BaseModel):
    step: int
    mean_reward: float
    pass_at_1: float
    pass_at_4: float
    policy_temp: float
