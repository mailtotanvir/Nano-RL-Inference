from hashlib import sha256

from fastapi.testclient import TestClient


def test_provider_requires_matching_bearer_key_when_key_hashes_are_configured() -> None:
    from mini_stack.provider.app import create_app
    from mini_stack.schemas.registry import ModelRegistryEntry

    app = create_app(
        registry=(
            ModelRegistryEntry(
                model_id="nano-general", backend="vllm", model_name="qwen",
                capabilities=("chat",), hardware="cpu", context_length=1024,
                status="ready", routing_weight=1.0,
            ),
        ),
        api_key_hashes={sha256(b"local-test-key").hexdigest()},
    )
    client = TestClient(app)

    assert client.get("/v1/models").status_code == 401
    authorized = client.get("/v1/models", headers={"Authorization": "Bearer local-test-key"})
    assert authorized.status_code == 200
