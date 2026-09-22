"""Generic OpenAI-compatible remote backend for optional live validation.

Credentials and endpoint URLs are runtime-only environment variables. They are
never persisted in repository configuration or artifacts.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx

from .openai_compatible import BackendHealth, InferenceBackend


class RemoteOpenAIBackend(InferenceBackend):
    """Remote OpenAI-compatible backend with runtime-only configuration."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("REMOTE_OPENAI_MODEL", "remote-model")
        self._base_url = os.environ["REMOTE_OPENAI_BASE_URL"].rstrip("/")
        self._api_key = os.environ["REMOTE_OPENAI_API_KEY"]

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def health(self) -> BackendHealth:
        async with httpx.AsyncClient(timeout=30) as client:
            started = __import__("time").perf_counter()
            try:
                response = await client.get(f"{self._base_url}/models", headers=self._headers)
                healthy = response.status_code == 200
            except httpx.HTTPError:
                healthy = False
            latency_ms = (__import__("time").perf_counter() - started) * 1000
        return BackendHealth(healthy=healthy, latency_ms=latency_ms)

    async def complete(self, model: str, prompt: str) -> str:
        payload = {
            "model": model or self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 128,
        }
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions", headers=self._headers, json=payload
            )
            response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        payload = {
            "model": model or self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 128,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=90) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", headers=self._headers, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        yield line.removeprefix("data: ")
