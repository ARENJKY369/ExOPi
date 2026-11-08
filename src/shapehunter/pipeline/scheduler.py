"""Simple deterministic, caller-driven scheduling of batch inputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from shapehunter.pipeline.batch import run_batch


@dataclass(frozen=True)
class ScheduledBatch:
    """A batch to run at or after a specified time."""

    due: datetime
    inputs: tuple[Path, ...]
    output: Path


def run_due(jobs: list[ScheduledBatch], now: datetime) -> list[ScheduledBatch]:
    """Execute due jobs synchronously and return the not-yet-due jobs.

    The caller is responsible for persisting the returned list and invoking
    this function again; this module does not run a background daemon.
    """
    remaining = []
    for job in jobs:
        if job.due <= now:
            run_batch(list(job.inputs), job.output)
        else:
            remaining.append(job)
    return remaining
