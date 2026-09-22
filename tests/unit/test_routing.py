import pytest


def test_weighted_router_uses_only_ready_models_and_falls_back() -> None:
    from mini_stack.provider.routing import choose_model
    from mini_stack.schemas.registry import ModelRegistryEntry

    models = (
        ModelRegistryEntry(
            model_id="nano-general", backend="vllm", model_name="qwen", capabilities=("chat",),
            hardware="cpu", context_length=1024, status="ready", routing_weight=0.2,
        ),
        ModelRegistryEntry(
            model_id="nano-reasoner", backend="sglang", model_name="qwen-rl",
            capabilities=("chat",),
            hardware="cpu", context_length=1024, status="ready", routing_weight=0.8,
        ),
    )

    explicit = choose_model(models, requested="nano-general", random_value=0.9)
    assert explicit.model_id == "nano-general"
    routed = choose_model(models, requested="nano-auto", random_value=0.5)
    assert routed.model_id == "nano-reasoner"
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        choose_model(models, requested="missing", random_value=0.5)
