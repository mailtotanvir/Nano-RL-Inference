"""Typed model registry records."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistryEntry(BaseModel):
    """A model backend available to the provider router."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    model_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    backend: Literal["vllm", "sglang", "remote"]
    model_name: str = Field(min_length=1)
    capabilities: tuple[str, ...] = Field(min_length=1)
    hardware: str = Field(min_length=1)
    context_length: int = Field(gt=0)
    status: Literal["ready", "draining", "offline"]
    routing_weight: float = Field(ge=0, le=1)


class ModelRegistry(BaseModel):
    """Validated registry document used by provider routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"]
    models: tuple[ModelRegistryEntry, ...] = Field(min_length=1)
