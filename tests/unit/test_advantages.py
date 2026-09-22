import pytest


def test_rloo_advantages_are_leave_one_out_centered() -> None:
    from mini_stack.rl.advantages import rloo_advantages

    assert rloo_advantages((1.0, 2.0, 3.0)) == pytest.approx((-1.5, 0.0, 1.5))


def test_grpo_advantages_are_group_normalized() -> None:
    from mini_stack.rl.advantages import grpo_advantages

    advantages = grpo_advantages((1.0, 2.0, 3.0))

    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[1] == pytest.approx(0.0)


def test_ppo_ratio_clipping_limits_ratio() -> None:
    from mini_stack.rl.advantages import clip_ratio

    assert clip_ratio(1.4, epsilon=0.2) == pytest.approx(1.2)
    assert clip_ratio(0.5, epsilon=0.2) == pytest.approx(0.8)
