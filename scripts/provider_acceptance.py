#!/usr/bin/env python3
"""Execute a live provider acceptance request through a generic remote backend."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mini_stack.backends.remote_openai import RemoteOpenAIBackend
from mini_stack.provider.app import create_app
from mini_stack.schemas.registry import ModelRegistryEntry


def main() -> None:
    registry = (
        ModelRegistryEntry(
            model_id="nano-remote",
            backend="remote",
            model_name="remote-model",
            capabilities=("chat", "completion", "streaming"),
            hardware="managed",
            context_length=128000,
            status="ready",
            routing_weight=1.0,
        ),
    )
    app = create_app(registry=registry, backends={"remote": RemoteOpenAIBackend()})
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "nano-auto",
                "messages": [
                    {"role": "user", "content": "What is 17 + 25? Reply with only the answer."}
                ],
            },
        )
        health = client.get("/health")
        usage = client.get("/usage?model=nano-remote")
    payload = {"completion": response.json(), "health": health.json(), "usage": usage.json()}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
