"""Append-only experiment metric recording."""

from __future__ import annotations

from pathlib import Path

from mini_stack.schemas.experiment import ExperimentMetric
from mini_stack.schemas.serialization import append_jsonl


class MetricRecorder:
    """Persist validated metrics as one durable JSON object per line."""

    def __init__(self, target: Path) -> None:
        self._target = target

    def record(self, metric: ExperimentMetric) -> None:
        append_jsonl(self._target, metric)

    def records(self) -> int:
        if not self._target.exists():
            return 0
        return len(self._target.read_text(encoding="utf-8").splitlines())
