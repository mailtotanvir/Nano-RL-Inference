"""Conservative deterministic exact-match reward."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import cast

_BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\s*/\s*[-+]?\d[\d,]*)?")


def normalize_answer(value: str) -> str:
    """Extract and canonicalize the final integer or rational answer without evaluation."""

    text = value.strip()
    boxed = _BOXED_PATTERN.findall(text)
    if boxed:
        text = boxed[-1]
    matches = _NUMBER_PATTERN.findall(text)
    if not matches:
        return " ".join(text.lower().split())
    candidate = cast(str, matches[-1]).replace(",", "").replace(" ", "")
    try:
        return str(Fraction(candidate))
    except (ValueError, ZeroDivisionError):
        return candidate


def exact_match_reward(completion: str, expected_answer: str) -> float:
    """Return one for canonical exact match and zero otherwise."""

    return float(normalize_answer(completion) == normalize_answer(expected_answer))
