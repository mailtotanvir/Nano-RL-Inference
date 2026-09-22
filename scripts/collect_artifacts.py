#!/usr/bin/env python3
"""Create a SHA-256 inventory for publication evidence and selected artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("artifacts")
OUT = ROOT / "manifests" / "artifact-inventory.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path != OUT:
            records.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "manifest": str(OUT)}))


if __name__ == "__main__":
    main()
