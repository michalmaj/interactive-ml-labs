"""Tests for Decision Tree Splitter explanation helpers."""

from decision_tree_splitter.challenge import DecisionTreeChallengeResult
from decision_tree_splitter.explanation import build_explanation_text
from ml_lab_core import AlgorithmSnapshot

MODE_AUTO_TREE = "auto_tree"
MODE_MANUAL_SPLIT = "manual_split"


def _snapshot(
    *,
    accuracy: float = 0.75,
    max_depth: int = 2,
    leaf_count: int = 4,
) -> AlgorithmSnapshot:
    """Create a minimal snapshot for explanation tests."""
    return AlgorithmSnapshot(
        iteration=1,
        status="fitted",
        metrics={
            "training_accuracy": accuracy,
            "max_depth": max_depth,
            "leaf_count": leaf_count,
        },
        visual_state={},
        done=True,
    )


def _challenge_result(*, success: bool) -> DecisionTreeChallengeResult:
    """Create a minimal challenge result for explanation tests."""
    return DecisionTreeChallengeResult(
        status="success" if success else "failed",
        dataset_kind="xor",
        target_accuracy=0.95,
        max_allowed_depth=2,
        accuracy=1.0 if success else 0.5,
        max_depth=2,
        actual_depth=2,
        node_count=7,
        leaf_count=4,
        message="Challenge completed: the tree is accurate enough and simple enough.",
    )


def test_auto_mode_returns_challenge_message_when_successful() -> None:
    """Successful challenge should be highlighted in auto mode."""
    text = build_explanation_text(
        _snapshot(accuracy=1.0),
        mode=MODE_AUTO_TREE,
        manual_error=None,
        challenge_result=_challenge_result(success=True),
    )

    assert text == "Challenge completed: the tree is accurate enough and simple enough."


def test_auto_mode_explains_failed_challenge() -> None:
    """Failed challenge should include model complexity and target accuracy."""
    text = build_explanation_text(
        _snapshot(
            accuracy=0.5,
            max_depth=1,
            leaf_count=2,
        ),
        mode=MODE_AUTO_TREE,
        manual_error=None,
        challenge_result=_challenge_result(success=False),
    )

    assert text == (
        "Auto mode: recursive axis-aligned splits. "
        "max_depth=1, leaves=2, accuracy=0.50. "
        "failed: target accuracy=0.95."
    )


def test_manual_mode_explains_valid_manual_split() -> None:
    """Valid manual split should encourage comparing impurity and gain."""
    text = build_explanation_text(
        _snapshot(),
        mode=MODE_MANUAL_SPLIT,
        manual_error=None,
        challenge_result=_challenge_result(success=False),
    )

    assert text == "Manual mode: move the split and compare impurity/gain with your intuition."


def test_manual_mode_explains_invalid_manual_split() -> None:
    """Invalid manual split should explain that children are not useful."""
    text = build_explanation_text(
        _snapshot(),
        mode=MODE_MANUAL_SPLIT,
        manual_error="split must create two non-empty children.",
        challenge_result=_challenge_result(success=False),
    )

    assert text == "Manual split is invalid because it does not create two useful children."
