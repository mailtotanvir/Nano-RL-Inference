from pathlib import Path

import pytest


def test_load_registry_validates_entries_from_yaml(tmp_path: Path) -> None:
    from mini_stack.schemas.config import load_model_registry

    path = tmp_path / "registry.yaml"
    path.write_text(
        """
schema_version: "1.0"
models:
  - model_id: nano-general
    backend: sglang
    model_name: Qwen/Qwen2.5-0.5B-Instruct
    capabilities: [chat]
    hardware: NVIDIA L4
    context_length: 32768
    status: ready
    routing_weight: 1.0
""".lstrip(),
        encoding="utf-8",
    )

    registry = load_model_registry(path)

    assert registry.schema_version == "1.0"
    assert registry.models[0].backend == "sglang"


def test_load_registry_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    from mini_stack.schemas.config import load_model_registry

    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: '1.0'\nmodels: []\nunexpected: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        load_model_registry(path)
