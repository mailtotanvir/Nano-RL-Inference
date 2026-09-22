#!/usr/bin/env python3
"""Create frozen template-disjoint arithmetic train/dev/test JSONL files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from mini_stack.data.generate import Template, generate_examples
from mini_stack.data.splits import build_template_splits

OUT = Path("data/processed/arithmetic_v1")
SPLITS = build_template_splits(
    train_templates=("addition", "subtraction"),
    dev_templates=("multiplication",),
    test_templates=("division",),
)
COUNTS = {"train": 256, "dev": 128, "test": 128}


def write_split(name: str, templates: tuple[str, ...], seed: int) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for offset, template in enumerate(templates):
        for example in generate_examples(
            template=cast(Template, template),
            count=COUNTS[name],
            seed=seed + offset,
            minimum=1,
            maximum=100,
        ):
            rows.append(
                {"prompt": example.prompt, "answer": example.answer, "template": example.template}
            )
    target = OUT / f"{name}.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    target.write_text(payload, encoding="utf-8")
    return {
        "path": str(target),
        "rows": len(rows),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "arithmetic_v1",
        "seed": 42,
        "splits": {
            "train": write_split("train", SPLITS.train, 42),
            "dev": write_split("dev", SPLITS.dev, 84),
            "test": write_split("test", SPLITS.test, 126),
        },
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
