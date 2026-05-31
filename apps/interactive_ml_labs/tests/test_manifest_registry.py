"""Tests for the unified app shell manifest registry."""

import pytest
from interactive_ml_labs import (
    DEMO_BY_ID,
    DEMO_MANIFESTS,
    LEVEL_MANIFESTS,
    LEVEL_NAMES,
    DemoManifest,
    LocalizedText,
    demos_for_level,
    levels_from_manifests,
    validate_demo_registry,
)
from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.knn_scene import create_knn_vote_map_scene
from interactive_ml_labs.logistic_scene import create_logistic_regression_scene


def test_registry_contains_current_demo_levels() -> None:
    """Manifest registry should expose levels dynamically."""
    assert levels_from_manifests() == (1, 2)
    assert 1 in LEVEL_NAMES
    assert 2 in LEVEL_NAMES
    assert {level.number for level in LEVEL_MANIFESTS} >= {1, 2, 3}


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
        assert manifest.create_scene is not None


def test_boosting_manifest_has_demo_specific_teaching_content() -> None:
    """Boosting manifest should describe the real demo, not only placeholder content."""
    manifest = DEMO_BY_ID["boosting_mistake_lab"]
    polish_text = " ".join(
        [
            manifest.summary.pl,
            *(objective.pl for objective in manifest.objectives),
            *(control.action.pl for control in manifest.controls),
        ],
    )

    assert "weak learners" in polish_text
    assert "train/test accuracy" in polish_text
    assert "generalization gap" in polish_text
    assert "confidence view" in polish_text
    assert "decision boundary" in polish_text
    assert len(manifest.objectives) == 3
    assert len(manifest.controls) >= 5
    assert manifest.create_scene is create_boosting_mistake_lab_scene


def test_gradient_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Gradient manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["gradient_descent_playground"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_gradient_descent_scene
    assert "gradient descent" in text
    assert "learning rate" in text
    assert "dataset noise" in text
    assert len(manifest.controls) >= 6


def test_knn_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """k-NN manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["knn_vote_map"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_knn_vote_map_scene
    assert "k-NN" in text
    assert "query point" in text
    assert "dataset noise" in text
    assert len(manifest.controls) >= 6


def test_logistic_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Logistic Regression manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["logistic_regression_boundary_lab"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_logistic_regression_scene
    assert "logistic regression" in text
    assert "learning rate" in text
    assert "threshold" in text
    assert "precision" in text
    assert "recall" in text
    assert len(manifest.controls) >= 7


def test_registry_contains_polish_diacritics() -> None:
    """Polish registry text should preserve diacritics."""
    polish_text = " ".join(
        [
            *(manifest.title.pl for manifest in DEMO_MANIFESTS),
            *(manifest.summary.pl for manifest in DEMO_MANIFESTS),
        ],
    )

    assert "ł" in polish_text
    assert "ó" in polish_text
    assert "ę" in polish_text


def test_registry_validates_default_manifests() -> None:
    """Default registry should pass consistency validation."""
    validate_demo_registry()


def test_registry_rejects_duplicate_demo_ids() -> None:
    """Registry validation should reject duplicate demo identifiers."""
    duplicate = DEMO_MANIFESTS[0]

    with pytest.raises(ValueError, match="Duplicate demo ids"):
        validate_demo_registry(demo_manifests=(duplicate, duplicate))


def test_registry_rejects_unknown_demo_level() -> None:
    """Registry validation should reject demos assigned to unknown levels."""
    invalid_demo = DemoManifest(
        id="unknown_level_demo",
        level=99,
        title=LocalizedText(en="Unknown", pl="Nieznany"),
        summary=LocalizedText(en="Unknown level", pl="Nieznany poziom"),
        objectives=DEMO_MANIFESTS[0].objectives,
        controls=DEMO_MANIFESTS[0].controls,
        create_scene=DEMO_MANIFESTS[0].create_scene,
        tags=("test",),
    )

    with pytest.raises(ValueError, match="unknown level"):
        validate_demo_registry(demo_manifests=(invalid_demo,))
