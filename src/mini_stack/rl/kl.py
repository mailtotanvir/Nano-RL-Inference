"""Numerically transparent KL estimators for the RL study."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _validate(log_policy: Sequence[float], log_reference: Sequence[float]) -> None:
    if not log_policy or len(log_policy) != len(log_reference):
        raise ValueError("Log-probability sequences must be non-empty and equally sized")


def _weighted_mean(log_policy: Sequence[float], values: Sequence[float]) -> float:
    return sum(math.exp(log_p) * value for log_p, value in zip(log_policy, values, strict=True))


def kl_k1(log_policy: Sequence[float], log_reference: Sequence[float]) -> float:
    """Exact forward KL expectation under the policy for a discrete distribution."""

    _validate(log_policy, log_reference)
    values = [p - q for p, q in zip(log_policy, log_reference, strict=True)]
    return _weighted_mean(log_policy, values)


def kl_k2(log_policy: Sequence[float], log_reference: Sequence[float]) -> float:
    """Second-order log-ratio estimator."""

    _validate(log_policy, log_reference)
    return _weighted_mean(
        log_policy, [0.5 * (p - q) ** 2 for p, q in zip(log_policy, log_reference, strict=True)]
    )


def kl_k3(log_policy: Sequence[float], log_reference: Sequence[float]) -> float:
    """Non-negative control-variate estimator based on exponentiated log ratio."""

    _validate(log_policy, log_reference)
    return _weighted_mean(
        log_policy,
        [math.exp(q - p) - 1 - (q - p) for p, q in zip(log_policy, log_reference, strict=True)],
    )
