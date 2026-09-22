from fractions import Fraction


def test_generation_is_deterministic_for_same_seed() -> None:
    from mini_stack.data.generate import generate_examples

    first = generate_examples(template="addition", count=3, seed=42, minimum=1, maximum=9)
    second = generate_examples(template="addition", count=3, seed=42, minimum=1, maximum=9)

    assert first == second
    assert [item.template for item in first] == ["addition", "addition", "addition"]


def test_split_by_template_has_no_template_overlap() -> None:
    from mini_stack.data.splits import build_template_splits

    splits = build_template_splits(
        train_templates=("addition", "subtraction"),
        dev_templates=("multiplication",),
        test_templates=("division",),
    )

    assert not (set(splits.train) & set(splits.dev))
    assert not (set(splits.train) & set(splits.test))
    assert not (set(splits.dev) & set(splits.test))


def test_exact_match_reward_normalizes_integer_fraction_and_boxed_answer() -> None:
    from mini_stack.rewards.exact_match import exact_match_reward, normalize_answer

    assert normalize_answer("  \\boxed{1,200} ") == "1200"
    assert normalize_answer(" 6 / 8 ") == str(Fraction(3, 4))
    assert exact_match_reward("The result is \\boxed{6/8}.", "3/4") == 1.0
    assert exact_match_reward("I cannot solve this.", "3/4") == 0.0
