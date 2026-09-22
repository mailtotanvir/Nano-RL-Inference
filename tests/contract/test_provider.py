from fastapi.testclient import TestClient


def test_provider_exposes_health_and_openai_models() -> None:
    from mini_stack.provider.app import create_app
    from mini_stack.schemas.registry import ModelRegistryEntry

    app = create_app(
        registry=(
            ModelRegistryEntry(
                model_id="nano-general", backend="vllm", model_name="qwen",
                capabilities=("chat",), hardware="cpu", context_length=1024,
                status="ready", routing_weight=1.0,
            ),
        )
    )
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "nano-general"
