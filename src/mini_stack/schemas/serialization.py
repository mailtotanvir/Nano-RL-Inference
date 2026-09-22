"""Canonical serialization for reproducible configs and append-only evidence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot canonically serialize {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return stable JSON suitable for config identity and evidence records."""

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a config independent of mapping insertion order."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def append_jsonl(target: Path, record: Mapping[str, Any] | BaseModel) -> None:
    """Durably append exactly one canonical JSON record to a JSONL file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{canonical_json(record)}\n".encode()
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
