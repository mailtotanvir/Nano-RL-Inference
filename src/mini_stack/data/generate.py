"""Deterministic arithmetic problem generation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from random import Random
from typing import Literal

Template = Literal["addition", "subtraction", "multiplication", "division"]


@dataclass(frozen=True)
class ArithmeticExample:
    template: Template
    prompt: str
    answer: str


def generate_examples(
    *, template: Template, count: int, seed: int, minimum: int, maximum: int
) -> tuple[ArithmeticExample, ...]:
    """Generate deterministic, self-contained arithmetic examples."""

    if count < 0 or minimum > maximum:
        raise ValueError("Invalid arithmetic generation bounds")
    random = Random(seed)
    examples: list[ArithmeticExample] = []
    for _ in range(count):
        left = random.randint(minimum, maximum)
        right = random.randint(minimum, maximum)
        if template == "addition":
            prompt, answer = f"What is {left} + {right}?", str(left + right)
        elif template == "subtraction":
            prompt, answer = f"What is {left} - {right}?", str(left - right)
        elif template == "multiplication":
            prompt, answer = f"What is {left} * {right}?", str(left * right)
        else:
            right = right or 1
            prompt = f"What is {left} / {right}?"
            answer = str(Fraction(left, right))
        examples.append(ArithmeticExample(template=template, prompt=prompt, answer=answer))
    return tuple(examples)
