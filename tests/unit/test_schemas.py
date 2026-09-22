from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def metric_payload() -> dict[str, object]:
    return {
        "run_id": "run-001",
        "timestamp": datetime(2026, 9, 20, tzinfo=UTC),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset": "arithmetic_v1",
        "algorithm": "sft",
        "seed": 42,
        "training_steps": 10,
        "wall_clock_seconds": 12.5,
        "gpu_type": "NVIDIA L4",
        "gpu_hours": 0.01,
        "peak_gpu_memory_gb": 2.5,
        "reward": 0.8,
        "evaluation_accuracy": 0.75,
        "response_length_tokens": 18.0,
        "token_throughput": 120.0,
        "kl": 0.02,
        "entropy": 1.2,
        "gradient_norm": 0.4,
        "loss": 0.3,
        "checkpoint_path": "artifacts/model/sft",
    }


def test_metric_record_round_trips_required_reproducibility_fields() -> None:
    from mini_stack.schemas.experiment import ExperimentMetric

    record = ExperimentMetric.model_validate(metric_payload())

    assert record.schema_version == "1.0"
    assert ExperimentMetric.model_validate_json(record.model_dump_json()) == record


def test_metric_record_rejects_missing_seed() -> None:
    from mini_stack.schemas.experiment import ExperimentMetric

    payload = metric_payload()
    payload.pop("seed")

    with pytest.raises(ValidationError, match="seed"):
        ExperimentMetric.model_validate(payload)


def test_model_registry_entry_requires_routable_backend_and_weight() -> None:
    from mini_stack.schemas.registry import ModelRegistryEntry

    entry = ModelRegistryEntry.model_validate(
        {
            "model_id": "nano-general",
            "backend": "vllm",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "capabilities": ["chat", "completion"],
            "hardware": "NVIDIA L4",
            "context_length": 32768,
            "status": "ready",
            "routing_weight": 0.7,
        }
    )

    assert entry.model_id == "nano-general"
    assert entry.routing_weight == pytest.approx(0.7)


def test_usage_event_requires_non_negative_latency_and_tokens() -> None:
    from mini_stack.schemas.usage import UsageEvent

    with pytest.raises(ValidationError, match="input_tokens"):
        UsageEvent.model_validate(
            {
                "request_id": "req-1",
                "model_id": "nano-general",
                "timestamp": datetime(2026, 9, 20, tzinfo=UTC),
                "input_tokens": -1,
                "output_tokens": 2,
                "latency_ms": 10.0,
                "ttft_ms": 4.0,
                "tpot_ms": 2.0,
                "queue_time_ms": 0.0,
                "error": None,
            }
        )
