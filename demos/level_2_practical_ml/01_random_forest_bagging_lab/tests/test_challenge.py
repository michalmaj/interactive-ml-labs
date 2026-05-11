"""Tests for Random Forest challenge mode."""

import pytest
from random_forest_bagging_lab import (
    ModelComparisonReport,
    ModelReportMetrics,
    RandomForestChallenge,
    RandomForestChallengeConfig,
)


def _report(
    *,
    forest_test_accuracy: float,
    forest_train_accuracy: float = 0.95,
    forest_tree_count: int = 25,
    baseline_test_accuracy: float = 0.80,
    winner: str = "random_forest",
) -> ModelComparisonReport:
    """Create a minimal model comparison report for challenge tests."""
    single_tree = ModelReportMetrics(
        name="single_tree",
        train_accuracy=0.90,
        test_accuracy=baseline_test_accuracy,
        generalization_gap=0.10,
        max_depth=2,
        node_count=7,
        leaf_count=4,
    )
    forest = ModelReportMetrics(
        name="random_forest",
        train_accuracy=forest_train_accuracy,
        test_accuracy=forest_test_accuracy,
        generalization_gap=forest_train_accuracy - forest_test_accuracy,
        max_depth=2,
        tree_count=forest_tree_count,
        mean_test_confidence=0.86,
    )

    return ModelComparisonReport(
        single_tree=single_tree,
        forest=forest,
        train_accuracy_delta=forest.train_accuracy - single_tree.train_accuracy,
        test_accuracy_delta=forest.test_accuracy - single_tree.test_accuracy,
        winner=winner,
        summary="Random forest comparison summary.",
    )


def test_random_forest_challenge_succeeds_when_targets_are_met() -> None:
    """Challenge should succeed when accuracy, tree count, and gap are acceptable."""
    challenge = RandomForestChallenge()

    result = challenge.evaluate(
        _report(
            forest_test_accuracy=0.92,
            forest_train_accuracy=0.96,
            forest_tree_count=25,
        ),
    )

    assert result.status == "success"
    assert result.success is True
    assert result.failed is False
    assert result.forest_test_accuracy == pytest.approx(0.92)
    assert result.forest_tree_count == 25


def test_random_forest_challenge_fails_when_test_accuracy_is_too_low() -> None:
    """Challenge should fail when forest test accuracy is too low."""
    challenge = RandomForestChallenge()

    result = challenge.evaluate(
        _report(
            forest_test_accuracy=0.70,
            forest_train_accuracy=0.80,
            forest_tree_count=25,
        ),
    )

    assert result.status == "failed"
    assert result.failed is True
    assert "test accuracy is too low" in result.message


def test_random_forest_challenge_fails_when_too_many_trees_are_used() -> None:
    """Challenge should fail when tree count is above the limit."""
    challenge = RandomForestChallenge()

    result = challenge.evaluate(
        _report(
            forest_test_accuracy=0.95,
            forest_train_accuracy=0.97,
            forest_tree_count=29,
        ),
    )

    assert result.status == "failed"
    assert "too many trees" in result.message


def test_random_forest_challenge_fails_when_gap_is_too_high() -> None:
    """Challenge should fail when generalization gap is too high."""
    challenge = RandomForestChallenge(
        RandomForestChallengeConfig(max_generalization_gap=0.05),
    )

    result = challenge.evaluate(
        _report(
            forest_test_accuracy=0.90,
            forest_train_accuracy=1.00,
            forest_tree_count=25,
        ),
    )

    assert result.status == "failed"
    assert "generalization gap" in result.message


def test_random_forest_challenge_rejects_report_without_forest_tree_count() -> None:
    """Forest tree count is required for challenge evaluation."""
    report = _report(forest_test_accuracy=0.95)
    broken_forest = ModelReportMetrics(
        name=report.forest.name,
        train_accuracy=report.forest.train_accuracy,
        test_accuracy=report.forest.test_accuracy,
        generalization_gap=report.forest.generalization_gap,
        max_depth=report.forest.max_depth,
        tree_count=None,
        mean_test_confidence=report.forest.mean_test_confidence,
    )
    broken_report = ModelComparisonReport(
        single_tree=report.single_tree,
        forest=broken_forest,
        train_accuracy_delta=report.train_accuracy_delta,
        test_accuracy_delta=report.test_accuracy_delta,
        winner=report.winner,
        summary=report.summary,
    )
    challenge = RandomForestChallenge()

    with pytest.raises(ValueError, match="tree_count"):
        challenge.evaluate(broken_report)


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            RandomForestChallengeConfig(target_test_accuracy=0.0),
            "target_test_accuracy",
        ),
        (
            RandomForestChallengeConfig(target_test_accuracy=1.1),
            "target_test_accuracy",
        ),
        (
            RandomForestChallengeConfig(max_tree_count=0),
            "max_tree_count",
        ),
        (
            RandomForestChallengeConfig(max_generalization_gap=-0.1),
            "max_generalization_gap",
        ),
    ],
)
def test_random_forest_challenge_rejects_invalid_config(
    config: RandomForestChallengeConfig,
    expected_message: str,
) -> None:
    """Invalid challenge configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        RandomForestChallenge(config)
