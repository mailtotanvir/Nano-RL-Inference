import json
from datetime import UTC, datetime
from pathlib import Path


def test_canonical_config_hash_is_independent_of_mapping_order() -> None:
    from mini_stack.schemas.serialization import config_sha256

    first = {"seed": 42, "model": {"name": "qwen", "rank": 8}}
    second = {"model": {"rank": 8, "name": "qwen"}, "seed": 42}

    assert config_sha256(first) == config_sha256(second)


def test_append_jsonl_writes_one_valid_record_per_line(tmp_path: Path) -> None:
    from mini_stack.schemas.serialization import append_jsonl

    target = tmp_path / "metrics.jsonl"
    record = {"run_id": "run-1", "timestamp": datetime(2026, 9, 20, tzinfo=UTC)}

    append_jsonl(target, record)
    append_jsonl(target, {"run_id": "run-2"})

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == ["run-1", "run-2"]
    assert json.loads(lines[0])["timestamp"] == "2026-09-20T00:00:00+00:00"
