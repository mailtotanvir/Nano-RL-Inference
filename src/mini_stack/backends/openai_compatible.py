"""Async OpenAI-compatible backend adapters for vLLM and SGLang servers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx


@dataclass(frozen=True)
class BackendHealth:
    healthy: bool
    latency_ms: float


class InferenceBackend(Protocol):
    async def health(self) -> BackendHealth: ...
    async def complete(self, model: str, prompt: str) -> str: ...
    def stream(self, model: str, prompt: str) -> AsyncIterator[str]: ...


class OpenAIBackend:
    """Adapter used for both vLLM and SGLang OpenAI-compatible servers."""

    def __init__(self, base_url: str, timeout_seconds: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    async def health(self) -> BackendHealth:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            started = __import__("time").perf_counter()
            try:
                response = await client.get(f"{self._base_url}/health")
                healthy = response.status_code == 200
            except httpx.HTTPError:
                healthy = False
            elapsed = (__import__("time").perf_counter() - started) * 1000
        return BackendHealth(healthy=healthy, latency_ms=elapsed)

    async def complete(self, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": 128},
            )
            response.raise_for_status()
        return str(response.json()["choices"][0]["text"])

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/completions",
                json={"model": model, "prompt": prompt, "max_tokens": 128, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        yield line.removeprefix("data: ")


class VLLMBackend(OpenAIBackend):
    pass


class SGLangBackend(OpenAIBackend):
    pass
