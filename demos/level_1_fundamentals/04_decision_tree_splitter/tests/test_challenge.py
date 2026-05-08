"""Tests for Decision Tree Splitter challenge mode."""

import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    DecisionTreeChallenge,
    DecisionTreeChallengeConfig,
)
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    accuracy: float,
    max_depth: int,
    actual_depth: int = 1,
    node_count: int = 3,
    leaf_count: int = 2,
) -> AlgorithmSnapshot:
    """Create a minimal tree snapshot for challenge tests."""
    return AlgorithmSnapshot(
        iteration=node_count,
        status="fitted",
        metrics={
            "training_accuracy": accuracy,
            "max_depth": max_depth,
            "actual_depth": actual_depth,
            "node_count": node_count,
            "leaf_count": leaf_count,
        },
        visual_state={},
        done=True,
    )


def test_axis_aligned_challenge_succeeds_for_accurate_depth_one_tree() -> None:
    """Axis-aligned challenge should reward a simple accurate tree."""
    challenge = DecisionTreeChallenge()

    result = challenge.evaluate(
        snapshot=_snapshot(accuracy=1.0, max_depth=1),
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
    )

    assert result.status == "success"
    assert result.success is True
    assert result.failed is False
    assert result.max_allowed_depth == 1
    assert result.accuracy == pytest.approx(1.0)


def test_axis_aligned_challenge_fails_when_tree_is_too_deep() -> None:
    """Axis-aligned challenge should fail if tree is unnecessarily deep."""
    challenge = DecisionTreeChallenge()

    result = challenge.evaluate(
        snapshot=_snapshot(accuracy=1.0, max_depth=2),
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
    )

    assert result.status == "failed"
    assert result.success is False
    assert result.failed is True
    assert "max_depth is too high" in result.message


def test_xor_challenge_succeeds_for_accurate_depth_two_tree() -> None:
    """XOR challenge should allow depth two."""
    challenge = DecisionTreeChallenge()

    result = challenge.evaluate(
        snapshot=_snapshot(
            accuracy=1.0,
            max_depth=2,
            actual_depth=2,
            node_count=7,
            leaf_count=4,
        ),
        dataset_kind=DATASET_KIND_XOR,
    )

    assert result.status == "success"
    assert result.max_allowed_depth == 2
    assert result.node_count == 7
    assert result.leaf_count == 4


def test_challenge_rejects_unknown_dataset_kind() -> None:
    """Unknown dataset kinds should fail clearly."""
    challenge = DecisionTreeChallenge()

    with pytest.raises(ValueError, match="Unsupported dataset kind"):
        challenge.evaluate(
            snapshot=_snapshot(accuracy=1.0, max_depth=1),
            dataset_kind="unknown",
        )


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            DecisionTreeChallengeConfig(target_accuracy=0.0),
            "target_accuracy",
        ),
        (
            DecisionTreeChallengeConfig(target_accuracy=1.1),
            "target_accuracy",
        ),
        (
            DecisionTreeChallengeConfig(axis_aligned_max_depth=0),
            "axis_aligned_max_depth",
        ),
        (
            DecisionTreeChallengeConfig(xor_max_depth=0),
            "xor_max_depth",
        ),
    ],
)
def test_challenge_rejects_invalid_config(
    config: DecisionTreeChallengeConfig,
    expected_message: str,
) -> None:
    """Invalid challenge configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        DecisionTreeChallenge(config)
