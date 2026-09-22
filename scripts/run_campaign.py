#!/usr/bin/env python3
"""Create or advance a local resumable campaign manifest without cloud side effects."""

from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from mini_stack.campaign import CampaignState, load_state, save_state
from mini_stack.schemas.serialization import config_sha256


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--advance")
    args = parser.parse_args()
    target = Path("artifacts/manifests") / f"{args.run_id}.json"
    now = datetime.now(UTC)
    if target.exists():
        state = load_state(target)
    else:
        state = CampaignState.create(
            run_id=args.run_id,
            git_commit=commit(),
            config_hash=config_sha256({"config": args.config.read_text(encoding="utf-8")}),
            now=now,
        )
    if args.advance:
        state = state.advance(args.advance, now=now)
    save_state(target, state)
    print(target)


if __name__ == "__main__":
    main()
