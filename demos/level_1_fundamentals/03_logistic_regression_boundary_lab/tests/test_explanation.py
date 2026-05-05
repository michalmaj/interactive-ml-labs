"""Tests for Logistic Regression Boundary Lab explanation helpers."""

from logistic_regression_boundary_lab.challenge import PrecisionRecallChallengeResult
from logistic_regression_boundary_lab.explanation import build_explanation_lines
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    iteration: int,
    loss: float = 0.5,
    accuracy: float = 0.8,
    precision: float = 0.75,
    recall: float = 0.9,
    threshold: float = 0.5,
    false_positive: int = 3,
    false_negative: int = 2,
) -> AlgorithmSnapshot:
    """Create a minimal snapshot for explanation tests."""
    return AlgorithmSnapshot(
        iteration=iteration,
        metrics={
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "threshold": threshold,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        visual_state={},
    )


def _challenge_result(*, success: bool) -> PrecisionRecallChallengeResult:
    """Create a minimal challenge result for explanation tests."""
    return PrecisionRecallChallengeResult(
        status="success" if success else "failed",
        target_precision=0.8,
        target_recall=0.9,
        accuracy=0.85,
        precision=0.82 if success else 0.70,
        recall=0.92 if success else 0.75,
        true_positive=18,
        true_negative=16,
        false_positive=4,
        false_negative=2,
        sample_count=40,
        message="Challenge message.",
    )


def test_explanation_before_training_with_failed_challenge() -> None:
    """Initial explanation should describe probability and thresholding."""
    lines = build_explanation_lines(
        _snapshot(iteration=0),
        _challenge_result(success=False),
    )

    assert lines == (
        "The background shows probability of class_1 before thresholding.",
        "Train the model, then tune threshold to balance FP and FN.",
    )


def test_explanation_before_training_with_successful_challenge() -> None:
    """Initial explanation should mention challenge success if already satisfied."""
    lines = build_explanation_lines(
        _snapshot(iteration=0),
        _challenge_result(success=True),
    )

    assert lines == (
        "Challenge is already completed on hidden test data.",
        "Train step by step and observe how probability background changes.",
    )


def test_explanation_after_training_with_failed_challenge() -> None:
    """Explanation after training should include model metrics and FP/FN."""
    lines = build_explanation_lines(
        _snapshot(
            iteration=5,
            loss=0.321,
            accuracy=0.86,
            precision=0.74,
            recall=0.91,
            threshold=0.45,
            false_positive=5,
            false_negative=1,
        ),
        _challenge_result(success=False),
    )

    assert lines == (
        "Loss=0.3210, accuracy=0.86, precision=0.74, recall=0.91.",
        "Challenge not completed: threshold=0.45, FP=5, FN=1; tune threshold.",
    )


def test_explanation_after_training_with_successful_challenge() -> None:
    """Explanation should mention success when hidden-test targets are satisfied."""
    lines = build_explanation_lines(
        _snapshot(
            iteration=10,
            loss=0.25,
            accuracy=0.9,
            precision=0.85,
            recall=0.95,
            threshold=0.55,
            false_positive=2,
            false_negative=1,
        ),
        _challenge_result(success=True),
    )

    assert lines == (
        "Loss=0.2500, accuracy=0.90, precision=0.85, recall=0.95.",
        "Challenge success: threshold=0.55, FP=2, FN=1.",
    )
