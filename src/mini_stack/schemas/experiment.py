"""Versioned Pydantic schemas for reproducible experiment evidence."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SchemaVersion = Literal["1.0"]


class ExperimentMetric(BaseModel):
    """One append-only metric record from a training or evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = "1.0"
    run_id: str = Field(min_length=1)
    timestamp: datetime
    model: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    algorithm: str = Field(min_length=1)
    seed: int = Field(ge=0)
    training_steps: int = Field(ge=0)
    wall_clock_seconds: float = Field(ge=0)
    gpu_type: str = Field(min_length=1)
    gpu_hours: float = Field(ge=0)
    peak_gpu_memory_gb: float = Field(ge=0)
    reward: float
    evaluation_accuracy: float = Field(ge=0, le=1)
    response_length_tokens: float = Field(ge=0)
    token_throughput: float = Field(ge=0)
    kl: float = Field(ge=0)
    entropy: float = Field(ge=0)
    gradient_norm: float = Field(ge=0)
    loss: float
    checkpoint_path: str = Field(min_length=1)


class CampaignManifest(BaseModel):
    """Immutable identity and state for a resumable campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = "1.0"
    run_id: str = Field(min_length=1)
    git_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    stage: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    completed_stages: tuple[str, ...] = ()
