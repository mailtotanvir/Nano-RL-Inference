#!/usr/bin/env python3
"""Fail closed when required pre-publication evidence is absent."""

from __future__ import annotations

from pathlib import Path

REQUIRED = (
    Path("reports/initial-training-results.md"),
    Path("reports/engine-smoke-benchmark.md"),
    Path("artifacts/metrics/sft-test.json"),
    Path("artifacts/metrics/grpo-test.json"),
    Path("artifacts/metrics/reinforce-test.json"),
    Path("artifacts/metrics/rloo-test.json"),
    Path("artifacts/metrics/ppo-test.json"),
    Path("artifacts/benchmarks/vllm-full.json"),
    Path("artifacts/benchmarks/sglang-full.json"),
    Path("artifacts/manifests/artifact-inventory.json"),
)


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.is_file()]
    if missing:
        raise SystemExit("Missing required evidence:\n" + "\n".join(missing))
    print("evidence verification passed")


if __name__ == "__main__":
    main()
