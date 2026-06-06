"""Toy reasoning tasks."""

from __future__ import annotations

from grpo.types import Task


def tasks() -> list[Task]:
    return [
        Task(tid="t-01", question="2+3", answer="5"),
        Task(tid="t-02", question="7*6", answer="42"),
        Task(tid="t-03", question="12-7", answer="5"),
        Task(tid="t-04", question="100/4", answer="25"),
        Task(tid="t-05", question="11^2", answer="121"),
        Task(tid="t-06", question="9+9", answer="18"),
        Task(tid="t-07", question="5!", answer="120"),
        Task(tid="t-08", question="sqrt(81)", answer="9"),
    ]
