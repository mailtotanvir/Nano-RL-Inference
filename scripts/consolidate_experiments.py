#!/usr/bin/env python3
"""Consolidate executed experiment evidence into the required JSONL ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

RUNS = {
    "sft": "artifacts/metrics/sft-test.json",
    "grpo": "artifacts/metrics/grpo-test.json",
    "reinforce": "artifacts/metrics/reinforce-test.json",
    "rloo": "artifacts/metrics/rloo-test.json",
    "ppo": "artifacts/metrics/ppo-test.json",
}
OUT = Path("artifacts/metrics/experiments.jsonl")


def main() -> None:
    records = []
    for algorithm, source in RUNS.items():
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        records.append(
            {
                "schema_version": "1.0",
                "run_id": f"{algorithm}-20260921",
                "timestamp": datetime.now(UTC).isoformat(),
                "model": payload["model"],
                "dataset": "arithmetic_v1_template_disjoint",
                "algorithm": algorithm,
                "seed": 42,
                "training_steps": None,
                "wall_clock_seconds": None,
                "gpu_type": "NVIDIA L4",
                "gpu_hours": None,
                "peak_gpu_memory_gb": None,
                "reward": None,
                "evaluation_accuracy": payload["accuracy"],
                "response_length": None,
                "token_throughput": None,
                "kl": None,
                "entropy": None,
                "gradient_norm": None,
                "loss": None,
                "checkpoint_path": payload["model"],
                "evidence_path": source,
                "missing_metrics": [
                    "training_steps", "wall_clock_seconds", "gpu_hours", "peak_gpu_memory_gb",
                    "response_length", "token_throughput", "entropy", "gradient_norm",
                ],
            }
        )
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    OUT.write_text(content, encoding="utf-8")
    print(json.dumps({"records": len(records), "path": str(OUT)}))


if __name__ == "__main__":
    main()
