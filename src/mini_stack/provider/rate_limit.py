"""Small deterministic in-process token bucket for the single-node provider."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: int
    refill_per_second: float
    _tokens: float = field(init=False)
    _last_refill: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.refill_per_second < 0:
            raise ValueError("refill_per_second must be non-negative")
        self._tokens = float(self.capacity)

    def allow(self, *, now: float) -> bool:
        if self._last_refill is not None:
            elapsed = max(0.0, now - self._last_refill)
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_second)
        self._last_refill = now
        if self._tokens < 1:
            return False
        self._tokens -= 1
        return True
