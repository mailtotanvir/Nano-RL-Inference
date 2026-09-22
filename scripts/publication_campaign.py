#!/usr/bin/env python3
"""Publication-evidence campaign orchestrator with per-run JSONL lifecycle records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

METHOD_SCRIPT = {
    "sft": "scripts/train_sft.py",
    "reinforce": "scripts/train_reinforce.py",
    "ppo": "scripts/train_ppo.py",
    "rloo": "scripts/train_rloo.py",
    "grpo": "scripts/train_grpo.py",
    "grpo_k1": "scripts/train_grpo.py",
    "grpo_k3": "scripts/train_grpo.py",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = Path("artifacts/metrics/publication-campaign.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    config_hash = file_hash(args.config)
    for seed in config["seeds"]:
        for method in config["methods"]:
            output_dir = f"artifacts/publication/{seed}/{method}"
            command = ["python", METHOD_SCRIPT[method]]
            environment = {
                **os.environ,
                "MINI_STACK_SEED": str(seed),
                "MINI_STACK_METHOD": method,
                "MINI_STACK_OUTPUT_DIR": output_dir,
                "MINI_STACK_METRICS_PATH": f"artifacts/metrics/{method}-seed{seed}.json",
                "MINI_STACK_SFT_ADAPTER": f"artifacts/publication/{seed}/sft",
            }
            record = {
                "schema_version": "1.0",
                "run_id": f"{config['run_id']}-{method}-seed{seed}",
                "timestamp": datetime.now(UTC).isoformat(),
                "algorithm": method,
                "seed": seed,
                "config_hash": config_hash,
                "command": command,
                "status": "planned" if args.dry_run else "not_implemented",
            }
            out.write_text("", encoding="utf-8") if not out.exists() else None
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if not args.dry_run:
                started = time.perf_counter()
                result = subprocess.run(command, check=False, env=environment)
                record["wall_clock_seconds"] = time.perf_counter() - started
                record["returncode"] = result.returncode
                record["status"] = "completed" if result.returncode == 0 else "failed"
                if result.returncode == 0:
                    evaluation_env = {
                        **environment,
                        "ADAPTER_PATH": output_dir,
                        "EVAL_OUT": f"artifacts/metrics/{method}-seed{seed}-test.json",
                    }
                    evaluation = subprocess.run(
                        ["python", "scripts/evaluate_sft.py"], check=False, env=evaluation_env
                    )
                    record["evaluation_returncode"] = evaluation.returncode
                    record["evaluation_path"] = evaluation_env["EVAL_OUT"]
                    if evaluation.returncode != 0:
                        record["status"] = "evaluation_failed"
                with out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                if result.returncode != 0:
                    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
