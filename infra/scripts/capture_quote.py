#!/usr/bin/env python3
"""Write a timestamped human-readable preflight cost quote from explicit inputs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--max-hours", type=float, required=True)
    parser.add_argument("--zone", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "captured_at": datetime.now(UTC).isoformat(),
        "machine_type": "g2-standard-8",
        "accelerator": "NVIDIA L4 x1",
        "zone": args.zone,
        "hourly_rate_usd": args.hourly_rate,
        "max_hours": args.max_hours,
        "maximum_compute_usd": args.hourly_rate * args.max_hours,
        "source_url": args.source_url,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
