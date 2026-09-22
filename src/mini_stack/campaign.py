"""Resumable, strictly ordered campaign state independent of cloud execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

STAGES = (
    "preflight", "bootstrap_gpu", "dataset", "base_eval", "sft", "sft_eval", "reinforce",
    "ppo", "rloo", "grpo", "grpo_kl_ablation", "select_model", "serve_vllm",
    "benchmark_vllm", "serve_sglang", "benchmark_sglang", "provider_acceptance", "package",
    "sync_laptop", "sync_oci", "verify_two_copies", "stop_vm", "delete_gcp_resources",
    "verify_zero_resources", "reconcile_cost",
)


@dataclass(frozen=True)
class CampaignState:
    run_id: str
    git_commit: str
    config_hash: str
    stage: str
    completed_stages: tuple[str, ...]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls, *, run_id: str, git_commit: str, config_hash: str, now: datetime
    ) -> CampaignState:
        return cls(run_id, git_commit, config_hash, STAGES[0], (), now, now)

    def advance(self, next_stage: str, *, now: datetime) -> CampaignState:
        try:
            index = STAGES.index(self.stage)
        except ValueError as error:
            raise ValueError(f"Unknown current stage: {self.stage}") from error
        expected = STAGES[index + 1] if index + 1 < len(STAGES) else None
        if next_stage != expected:
            raise ValueError(f"Expected next stage {expected}, got {next_stage}")
        return CampaignState(
            self.run_id, self.git_commit, self.config_hash, next_stage,
            self.completed_stages + (self.stage,), self.created_at, now,
        )


def save_state(target: Path, state: CampaignState) -> None:
    """Atomically persist a campaign state without invoking cloud actions."""

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    payload["created_at"] = state.created_at.isoformat()
    payload["updated_at"] = state.updated_at.isoformat()
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def load_state(target: Path) -> CampaignState:
    """Restore a previously persisted local campaign state."""

    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["created_at"] = datetime.fromisoformat(payload["created_at"])
    payload["updated_at"] = datetime.fromisoformat(payload["updated_at"])
    payload["completed_stages"] = tuple(payload["completed_stages"])
    return CampaignState(**payload)
