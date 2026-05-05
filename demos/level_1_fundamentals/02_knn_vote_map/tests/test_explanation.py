"""Tests for k-NN Vote Map explanation helpers."""

from knn_vote_map.challenge import KNNAccuracyChallengeResult
from knn_vote_map.explanation import build_explanation_lines
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    has_prediction: bool,
    predicted_label: int | None = None,
    vote_counts: dict[int, int] | None = None,
) -> AlgorithmSnapshot:
    """Create a minimal snapshot for explanation tests."""
    metrics: dict[str, int | float | str | bool] = {
        "k": 5,
        "has_prediction": has_prediction,
    }
    visual_state: dict[str, object] = {}

    if predicted_label is not None:
        metrics["predicted_label"] = predicted_label

    if vote_counts is not None:
        visual_state["vote_counts"] = vote_counts

    return AlgorithmSnapshot(
        metrics=metrics,
        visual_state=visual_state,
    )


def _challenge_result(*, success: bool) -> KNNAccuracyChallengeResult:
    """Create a minimal challenge result for explanation tests."""
    return KNNAccuracyChallengeResult(
        status="success" if success else "failed",
        target_accuracy=0.9,
        accuracy=0.95 if success else 0.75,
        correct_count=19 if success else 15,
        sample_count=20,
        message="Challenge message.",
    )


def test_explanation_before_prediction_with_successful_challenge() -> None:
    """Initial explanation should mention hidden test success."""
    lines = build_explanation_lines(
        _snapshot(has_prediction=False),
        _challenge_result(success=True),
    )

    assert lines == (
        "Challenge completed on hidden test data.",
        "Click the map or press N to inspect local k-NN decisions.",
    )


def test_explanation_before_prediction_with_failed_challenge() -> None:
    """Initial explanation should suggest parameter tuning when challenge fails."""
    lines = build_explanation_lines(
        _snapshot(has_prediction=False),
        _challenge_result(success=False),
    )

    assert lines == (
        "Challenge is not completed on hidden test data yet.",
        "Try changing k, noise, or seed; click the map to inspect decisions.",
    )


def test_explanation_after_prediction_includes_votes() -> None:
    """Explanation after prediction should summarize neighbor votes."""
    lines = build_explanation_lines(
        _snapshot(
            has_prediction=True,
            predicted_label=1,
            vote_counts={0: 2, 1: 3},
        ),
        _challenge_result(success=True),
    )

    assert lines[0] == "Prediction: class_1. Neighbor votes: class_0: 2, class_1: 3."
    assert lines[1] == "Challenge success: test accuracy is 0.95 with target 0.90."


def test_explanation_after_prediction_without_votes_falls_back_to_k() -> None:
    """Explanation should remain useful even without vote details."""
    lines = build_explanation_lines(
        _snapshot(
            has_prediction=True,
            predicted_label=0,
            vote_counts=None,
        ),
        _challenge_result(success=False),
    )

    assert lines[0] == "The query point was classified as class_0 using k=5."
    assert lines[1] == "Challenge not completed: accuracy is 0.75; try tuning k or reducing noise."
