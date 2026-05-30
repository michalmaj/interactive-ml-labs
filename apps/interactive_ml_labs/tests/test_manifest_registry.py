"""Tests for the unified app shell manifest registry."""

from interactive_ml_labs import DEMO_MANIFESTS, LEVEL_NAMES, demos_for_level, levels_from_manifests


def test_registry_contains_current_demo_levels() -> None:
    """Manifest registry should expose levels dynamically."""
    assert levels_from_manifests() == (1, 2)
    assert 1 in LEVEL_NAMES
    assert 2 in LEVEL_NAMES


def test_registry_groups_demos_by_level() -> None:
    """Demo lookup should return only demos from the requested level."""
    level_one_demos = demos_for_level(1)
    level_two_demos = demos_for_level(2)

    assert {demo.level for demo in level_one_demos} == {1}
    assert {demo.level for demo in level_two_demos} == {2}
    assert len(level_one_demos) >= 4
    assert len(level_two_demos) >= 2


def test_manifests_have_required_teaching_content() -> None:
    """Every manifest should include content used by generated intro screens."""
    for manifest in DEMO_MANIFESTS:
        assert manifest.id
        assert manifest.title.en
        assert manifest.title.pl
        assert manifest.summary.en
        assert manifest.summary.pl
        assert manifest.objectives
        assert manifest.controls
        assert manifest.tags
