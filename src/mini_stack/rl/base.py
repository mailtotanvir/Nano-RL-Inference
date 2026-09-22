"""Framework-independent contract for online RL algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RolloutSample:
    prompt: str
    completion: str
    reward: float


class OnlineRLAlgorithm(ABC):
    """Common online-RL surface used by framework adapters and educational implementations."""

    @abstractmethod
    def rollout(self, policy: object, prompts: tuple[str, ...]) -> tuple[RolloutSample, ...]:
        """Sample completions from a policy."""

    def reward(self, samples: tuple[RolloutSample, ...]) -> tuple[float, ...]:
        """Return already-attached rewards by default."""

        return tuple(sample.reward for sample in samples)

    @abstractmethod
    def advantage(self, samples: tuple[RolloutSample, ...]) -> tuple[float, ...]:
        """Compute learning signal for the sampled completions."""

    @abstractmethod
    def loss(self, policy: object, reference: object, samples: tuple[RolloutSample, ...]) -> float:
        """Compute a scalar policy loss."""

    @abstractmethod
    def update(self, policy: object, loss: float) -> object:
        """Apply one policy update and return the updated policy."""

    @abstractmethod
    def evaluate(self, policy: object) -> float:
        """Return evaluation accuracy or another declared scalar metric."""
