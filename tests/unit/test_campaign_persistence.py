from datetime import UTC, datetime
from pathlib import Path


def test_campaign_state_can_be_saved_and_restored(tmp_path: Path) -> None:
    from mini_stack.campaign import CampaignState, load_state, save_state

    state = CampaignState.create(
        run_id="run-1", git_commit="abcdef1", config_hash="a" * 64,
        now=datetime(2026, 9, 20, tzinfo=UTC),
    )
    target = tmp_path / "run-1.json"

    save_state(target, state)

    assert load_state(target) == state
