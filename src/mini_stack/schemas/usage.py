"""Typed provider usage records."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UsageEvent(BaseModel):
    """One provider request outcome suitable for JSONL aggregation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    timestamp: datetime
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    ttft_ms: float = Field(ge=0)
    tpot_ms: float = Field(ge=0)
    queue_time_ms: float = Field(ge=0)
    error: str | None = None
