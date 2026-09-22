"""Request-level latency and throughput benchmark aggregation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import quantiles
from time import perf_counter


@dataclass(frozen=True)
class RequestMeasurement:
    latency_ms: float
    ttft_ms: float
    output_tokens: int


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) == 1:
        return values[0]
    return quantiles(values, n=100, method="inclusive")[int(percentile_value) - 1]


def summarize(measurements: list[RequestMeasurement]) -> dict[str, float]:
    if not measurements:
        raise ValueError("measurements must not be empty")
    latencies = [item.latency_ms for item in measurements]
    total_seconds = sum(latencies) / 1000
    return {
        "requests": float(len(measurements)),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "mean_ttft_ms": sum(item.ttft_ms for item in measurements) / len(measurements),
        "tokens_per_second": sum(item.output_tokens for item in measurements) / total_seconds,
    }


def measure(call: Callable[[], tuple[float, int]]) -> RequestMeasurement:
    started = perf_counter()
    ttft_ms, output_tokens = call()
    return RequestMeasurement(
        latency_ms=(perf_counter() - started) * 1000,
        ttft_ms=ttft_ms,
        output_tokens=output_tokens,
    )
