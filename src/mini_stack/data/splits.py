"""Template-disjoint data split validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateSplits:
    train: tuple[str, ...]
    dev: tuple[str, ...]
    test: tuple[str, ...]


def build_template_splits(
    *,
    train_templates: tuple[str, ...],
    dev_templates: tuple[str, ...],
    test_templates: tuple[str, ...],
) -> TemplateSplits:
    """Return splits only when no template family appears in more than one split."""

    splits = TemplateSplits(train_templates, dev_templates, test_templates)
    all_templates = splits.train + splits.dev + splits.test
    if len(all_templates) != len(set(all_templates)):
        raise ValueError("Template families must be split-disjoint")
    if not all_templates:
        raise ValueError("At least one template family is required")
    return splits
