from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from mini_stack.backends.openai_compatible import BackendHealth
from mini_stack.provider.app import create_app
from mini_stack.schemas.registry import ModelRegistryEntry


class FakeBackend:
    async def health(self) -> BackendHealth:
        return BackendHealth(healthy=True, latency_ms=2.0)

    async def complete(self, model: str, prompt: str) -> str:
        return f"backend:{model}:{prompt}"

    async def stream(self, model: str, prompt: str) -> AsyncIterator[str]:
        yield ' {"text":"A"}'
        yield ' {"text":"B"}'


def test_provider_routes_to_backend_streams_and_records_usage() -> None:
    registry = (
        ModelRegistryEntry(
            model_id="nano-general", backend="vllm", model_name="qwen", capabilities=("chat",),
            hardware="cpu", context_length=1024, status="ready", routing_weight=1.0,
        ),
    )
    app = create_app(registry=registry, backends={"vllm": FakeBackend()}, rate_limit_capacity=10)
    with TestClient(app) as client:
        completion = client.post("/v1/completions", json={"model": "nano-auto", "prompt": "hello"})
        stream = client.post(
            "/v1/completions",
            json={"model": "nano-general", "prompt": "hello", "stream": True},
        )
        health = client.get("/health")
        usage = client.get("/usage?model=nano-general")
        metrics = client.get("/metrics")

    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "backend:qwen:hello"
    assert "data:  {\"text\":\"A\"}" in stream.text
    assert health.json()["status"] == "ok"
    assert usage.json()["requests"] == 2
    assert "mini_stack_requests_total" in metrics.text
