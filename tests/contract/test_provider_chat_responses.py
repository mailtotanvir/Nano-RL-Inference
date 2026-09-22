from fastapi.testclient import TestClient


def test_provider_supports_chat_responses_and_usage_aggregation() -> None:
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

    chat = client.post(
        "/v1/chat/completions",
        json={"model": "nano-general", "messages": [{"role": "user", "content": "2+2"}]},
    )
    response = client.post(
        "/v1/responses", json={"model": "nano-general", "input": "3+3"}
    )
    usage = client.get("/usage?model=nano-general")

    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "answer:2+2"
    assert response.status_code == 200
    assert response.json()["output_text"] == "answer:3+3"
    assert usage.json()["requests"] == 2
