"""OpenAI-compatible provider with routing, rate limits, streaming, usage, and telemetry."""

from __future__ import annotations

import hmac
import random
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from hashlib import sha256
from time import perf_counter, time
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import Counter as PrometheusCounter
from pydantic import BaseModel, Field

from mini_stack.backends.openai_compatible import InferenceBackend
from mini_stack.provider.rate_limit import TokenBucket
from mini_stack.provider.routing import choose_model
from mini_stack.schemas.registry import ModelRegistryEntry

REQUESTS = PrometheusCounter("mini_stack_requests_total", "Provider requests", ["model", "outcome"])


class CompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    stream: bool = False


class ChatMessage(BaseModel):
    role: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=1)
    stream: bool = False


class ResponsesRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str = Field(min_length=1)


def create_app(
    *,
    registry: Sequence[ModelRegistryEntry],
    complete: Callable[[str], str] | None = None,
    backends: Mapping[str, InferenceBackend] | None = None,
    api_key_hashes: frozenset[str] = frozenset(),
    rate_limit_capacity: int = 60,
) -> FastAPI:
    """Create a provider with explicit, alias, weighted, and backend routing."""

    app = FastAPI(title="Mini AI Stack Provider", version="0.1.0")
    backend_map = dict(backends or {})
    buckets: dict[str, TokenBucket] = {}
    usage: Counter[str] = Counter()

    def require_auth(authorization: Annotated[str | None, Header()] = None) -> str:
        if not api_key_hashes:
            return "anonymous"
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.removeprefix("Bearer ")
        digest = sha256(token.encode()).hexdigest()
        if not any(hmac.compare_digest(digest, known) for known in api_key_hashes):
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        return digest

    def rate_limit(identity: str) -> None:
        bucket = buckets.setdefault(identity, TokenBucket(rate_limit_capacity, rate_limit_capacity / 60))
        if not bucket.allow(now=time()):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    async def run(model_name: str, prompt: str, identity: str) -> tuple[ModelRegistryEntry, str, dict[str, int]]:
        rate_limit(identity)
        try:
            model = choose_model(registry, requested=model_name, random_value=random.random())
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        started = perf_counter()
        if model.backend in backend_map:
            text = await backend_map[model.backend].complete(model.model_name, prompt)
        elif complete is not None:
            text = complete(prompt)
        else:
            raise HTTPException(status_code=503, detail="No healthy backend configured")
        usage[model.model_id] += 1
        REQUESTS.labels(model=model.model_id, outcome="success").inc()
        elapsed = (perf_counter() - started) * 1000
        tokens = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split())}
        tokens["total_tokens"] = tokens["prompt_tokens"] + tokens["completion_tokens"]
        app.state.last_latency_ms = elapsed
        return model, text, tokens

    @app.get("/health")
    async def health() -> dict[str, object]:
        states = {name: (await backend.health()).healthy for name, backend in backend_map.items()}
        if not states:
            return {"status": "ok"}
        return {"status": "ok" if all(states.values()) else "degraded", "backends": states}

    @app.get("/v1/models", dependencies=[Depends(require_auth)])
    def models() -> dict[str, object]:
        return {"object": "list", "data": [{"id": item.model_id, "object": "model", "owned_by": "mini-ai-stack"} for item in registry if item.status == "ready"]}

    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest, identity: str = Depends(require_auth)
    ) -> object:
        if request.stream:
            async def events() -> AsyncIterator[str]:
                rate_limit(identity)
                model = choose_model(registry, requested=request.model, random_value=random.random())
                backend = backend_map.get(model.backend)
                if backend is None:
                    raise HTTPException(status_code=503, detail="Streaming backend unavailable")
                async for chunk in backend.stream(model.model_name, request.prompt):
                    yield f"data: {chunk}\n\n"
                usage[model.model_id] += 1
                REQUESTS.labels(model=model.model_id, outcome="success").inc()
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        model, text, tokens = await run(request.model, request.prompt, identity)
        return {"id": f"cmpl_{uuid4().hex}", "object": "text_completion", "created": int(time()), "model": model.model_id, "choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": tokens}

    @app.post("/v1/chat/completions")
    async def chat(
        request: ChatCompletionRequest, identity: str = Depends(require_auth)
    ) -> dict[str, object]:
        prompt = "\n".join(message.content for message in request.messages)
        model, text, tokens = await run(request.model, prompt, identity)
        return {"id": f"chatcmpl_{uuid4().hex}", "object": "chat.completion", "created": int(time()), "model": model.model_id, "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": tokens}

    @app.post("/v1/responses")
    async def responses(
        request: ResponsesRequest, identity: str = Depends(require_auth)
    ) -> dict[str, object]:
        model, text, tokens = await run(request.model, request.input, identity)
        return {"id": f"resp_{uuid4().hex}", "object": "response", "created_at": int(time()), "model": model.model_id, "output_text": text, "usage": tokens}

    @app.get("/usage", dependencies=[Depends(require_auth)])
    def get_usage(model: str = Query(min_length=1)) -> dict[str, object]:
        return {"model": model, "requests": usage[model], "last_latency_ms": getattr(app.state, "last_latency_ms", None)}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        lines = [f'mini_stack_requests_total{{model="{model}"}} {count}' for model, count in usage.items()]
        return PlainTextResponse("\n".join(lines) + "\n")

    return app
