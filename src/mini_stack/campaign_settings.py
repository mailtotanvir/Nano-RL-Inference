"""Shared environment contract for reproducible GPU campaign entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSettings:
    seed: int
    output_dir: Path
    metrics_path: Path
    method: str


def settings(default_method: str, default_output: str) -> RunSettings:
    seed = int(os.environ.get("MINI_STACK_SEED", "42"))
    method = os.environ.get("MINI_STACK_METHOD", default_method)
    output = Path(os.environ.get("MINI_STACK_OUTPUT_DIR", default_output))
    metrics = Path(
        os.environ.get("MINI_STACK_METRICS_PATH", f"artifacts/metrics/{method}-seed{seed}.json")
    )
    return RunSettings(seed=seed, output_dir=output, metrics_path=metrics, method=method)
