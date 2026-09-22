import math

import pytest


def test_kl_estimators_are_zero_for_identical_log_probabilities() -> None:
    from mini_stack.rl.kl import kl_k1, kl_k2, kl_k3

    log_probs = (math.log(0.5), math.log(0.5))

    assert kl_k1(log_probs, log_probs) == pytest.approx(0.0)
    assert kl_k2(log_probs, log_probs) == pytest.approx(0.0)
    assert kl_k3(log_probs, log_probs) == pytest.approx(0.0)


def test_k1_matches_exact_kl_for_a_small_categorical_distribution() -> None:
    from mini_stack.rl.kl import kl_k1

    policy = (0.75, 0.25)
    reference = (0.5, 0.5)
    expected = sum(p * math.log(p / q) for p, q in zip(policy, reference, strict=True))

    policy_logs = tuple(map(math.log, policy))
    reference_logs = tuple(map(math.log, reference))
    assert kl_k1(policy_logs, reference_logs) == pytest.approx(expected)
