from datetime import UTC, datetime
from pathlib import Path


def test_metric_recorder_appends_valid_experiment_jsonl(tmp_path: Path) -> None:
    from mini_stack.evaluation.metrics import MetricRecorder
    from mini_stack.schemas.experiment import ExperimentMetric

    recorder = MetricRecorder(tmp_path / "metrics.jsonl")
    recorder.record(
        ExperimentMetric(
            run_id="run-1", timestamp=datetime(2026, 9, 20, tzinfo=UTC), model="qwen",
            dataset="arithmetic", algorithm="grpo", seed=42, training_steps=1,
            wall_clock_seconds=1.0, gpu_type="cpu", gpu_hours=0.0, peak_gpu_memory_gb=0.0,
            reward=1.0, evaluation_accuracy=1.0, response_length_tokens=4.0,
            token_throughput=5.0, kl=0.0, entropy=0.0, gradient_norm=0.0, loss=0.0,
            checkpoint_path="artifacts/model",
        )
    )

    assert recorder.records() == 1
    assert "run-1" in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8")


def test_common_rl_contract_can_be_implemented_without_framework_dependency() -> None:
    from mini_stack.rl.base import OnlineRLAlgorithm, RolloutSample

    class ToyAlgorithm(OnlineRLAlgorithm):
        def rollout(self, policy: object, prompts: tuple[str, ...]) -> tuple[RolloutSample, ...]:
            return tuple(
                RolloutSample(prompt=prompt, completion="1", reward=1.0) for prompt in prompts
            )

        def advantage(self, samples: tuple[RolloutSample, ...]) -> tuple[float, ...]:
            return tuple(sample.reward for sample in samples)

        def loss(
            self, policy: object, reference: object, samples: tuple[RolloutSample, ...]
        ) -> float:
            return -sum(sample.reward for sample in samples)

        def update(self, policy: object, loss: float) -> object:
            return policy

        def evaluate(self, policy: object) -> float:
            return 1.0

    algorithm = ToyAlgorithm()
    samples = algorithm.rollout(object(), ("a", "b"))
    assert algorithm.advantage(samples) == (1.0, 1.0)
    assert algorithm.loss(object(), object(), samples) == -2.0
