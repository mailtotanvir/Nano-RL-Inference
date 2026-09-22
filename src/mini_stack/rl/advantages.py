"""Framework-independent learning-signal primitives for policy-gradient comparisons."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _require_group(values: Sequence[float], minimum: int = 1) -> None:
    if len(values) < minimum:
        raise ValueError(f"Expected at least {minimum} values")


def rloo_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Return leave-one-out advantages for one prompt's sampled completions."""

    _require_group(rewards, minimum=2)
    total = sum(rewards)
    size = len(rewards)
    return tuple(reward - (total - reward) / (size - 1) for reward in rewards)


def grpo_advantages(rewards: Sequence[float], epsilon: float = 1e-8) -> tuple[float, ...]:
    """Return mean-centered, standard-deviation-normalized group advantages."""

    _require_group(rewards)
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance) + epsilon
    return tuple((reward - mean) / scale for reward in rewards)


def clip_ratio(ratio: float, epsilon: float) -> float:
    """Apply PPO's symmetric probability-ratio clip."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    return min(max(ratio, 1 - epsilon), 1 + epsilon)
