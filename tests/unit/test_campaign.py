from datetime import UTC, datetime

import pytest


def test_campaign_manifest_only_advances_to_next_declared_stage() -> None:
    from mini_stack.campaign import CampaignState

    state = CampaignState.create(
        run_id="run-1", git_commit="abcdef1", config_hash="a" * 64,
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )

    advanced = state.advance("bootstrap_gpu", now=datetime(2026, 9, 20, 0, 1, tzinfo=UTC))

    assert advanced.stage == "bootstrap_gpu"
    assert advanced.completed_stages == ("preflight",)
    with pytest.raises(ValueError, match="Expected next stage"):
        advanced.advance("grpo", now=datetime(2026, 9, 20, 0, 2, tzinfo=UTC))
