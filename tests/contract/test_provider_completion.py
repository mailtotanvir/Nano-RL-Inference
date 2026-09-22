from fastapi.testclient import TestClient


def test_provider_returns_openai_completion_and_usage() -> None:
    from mini_stack.provider.app import create_app
    from mini_stack.schemas.registry import ModelRegistryEntry

    app = create_app(
        registry=(
            ModelRegistryEntry(
                model_id="nano-general", backend="vllm", model_name="qwen",
                capabilities=("chat", "completion"), hardware="cpu", context_length=1024,
                status="ready", routing_weight=1.0,
            ),
        ),
        complete=lambda prompt: f"answer:{prompt}",
    )
    client = TestClient(app)

    response = client.post("/v1/completions", json={"model": "nano-general", "prompt": "2+2"})

    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "answer:2+2"
    assert response.json()["usage"]["total_tokens"] > 0
