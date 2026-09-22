"""Loading and validation for versioned YAML configuration documents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .registry import ModelRegistry


def load_model_registry(path: Path) -> ModelRegistry:
    """Load a registry YAML document and reject malformed or unknown fields."""

    parsed: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Registry at {path} must be a mapping")
    return ModelRegistry.model_validate(parsed)
