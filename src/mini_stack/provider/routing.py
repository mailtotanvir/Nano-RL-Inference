"""Deterministic, testable model resolution and weighted routing."""

from __future__ import annotations

from collections.abc import Sequence

from mini_stack.schemas.registry import ModelRegistryEntry


def choose_model(
    models: Sequence[ModelRegistryEntry], *, requested: str, random_value: float
) -> ModelRegistryEntry:
    """Resolve explicit models or the `nano-auto` weighted alias among ready entries."""

    ready = tuple(model for model in models if model.status == "ready")
    if requested != "nano-auto":
        for model in ready:
            if model.model_id == requested:
                return model
        raise ValueError(f"Unknown or unavailable model: {requested}")
    if not ready:
        raise ValueError("Unknown or unavailable model: nano-auto")
    if not 0 <= random_value < 1:
        raise ValueError("random_value must be in [0, 1)")
    total = sum(model.routing_weight for model in ready)
    if total <= 0:
        raise ValueError("Ready models require a positive total routing weight")
    threshold = random_value * total
    cumulative = 0.0
    for model in ready:
        cumulative += model.routing_weight
        if threshold < cumulative:
            return model
    return ready[-1]
