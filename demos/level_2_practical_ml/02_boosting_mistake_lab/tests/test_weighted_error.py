"""Tests for weighted error utilities."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    WeightedErrorResult,
    evaluate_weighted_predictions,
    weighted_accuracy_score,
    weighted_error_score,
)


def test_evaluate_weighted_predictions_returns_result() -> None:
    """Weighted prediction evaluation should return a result object."""
    result = evaluate_weighted_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert isinstance(result, WeightedErrorResult)


def test_evaluate_weighted_predictions_computes_weighted_error() -> None:
    """Weighted error should sum weights of misclassified samples."""
    result = evaluate_weighted_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert result.weighted_error == pytest.approx(0.6)
    assert result.mistake_weight_sum == pytest.approx(0.6)


def test_evaluate_weighted_predictions_computes_weighted_accuracy() -> None:
    """Weighted accuracy should sum weights of correctly classified samples."""
    result = evaluate_weighted_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert result.weighted_accuracy == pytest.approx(0.4)
    assert result.correct_weight_sum == pytest.approx(0.4)


def test_evaluate_weighted_predictions_returns_masks() -> None:
    """Evaluation should expose mistake and correct masks."""
    result = evaluate_weighted_predictions(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    np.testing.assert_array_equal(result.mistake_mask, np.array([False, True, False, True]))
    np.testing.assert_array_equal(result.correct_mask, np.array([True, False, True, False]))


def test_weighted_error_score_returns_only_error() -> None:
    """Convenience helper should return only weighted error."""
    error = weighted_error_score(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert error == pytest.approx(0.6)


def test_weighted_accuracy_score_returns_only_accuracy() -> None:
    """Convenience helper should return only weighted accuracy."""
    accuracy = weighted_accuracy_score(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert accuracy == pytest.approx(0.4)


def test_weighted_error_is_zero_for_perfect_predictions() -> None:
    """Perfect predictions should have zero weighted error."""
    error = weighted_error_score(
        y_true=np.array([0, 1, 1]),
        y_pred=np.array([0, 1, 1]),
        sample_weights=np.array([0.2, 0.3, 0.5]),
    )

    assert error == pytest.approx(0.0)


def test_weighted_error_is_one_when_all_predictions_are_wrong() -> None:
    """Completely wrong predictions should have weighted error equal to 1.0."""
    error = weighted_error_score(
        y_true=np.array([0, 1, 1]),
        y_pred=np.array([1, 0, 0]),
        sample_weights=np.array([0.2, 0.3, 0.5]),
    )

    assert error == pytest.approx(1.0)


@pytest.mark.parametrize(
    "y_true, y_pred, sample_weights, expected_message",
    [
        (
            np.array([[0, 1]]),
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            "y_true must be a one-dimensional array",
        ),
        (
            np.array([0, 1]),
            np.array([[0, 1]]),
            np.array([0.5, 0.5]),
            "y_pred must be a one-dimensional array",
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            "y_true must contain integers",
        ),
        (
            np.array([0, 1]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            "y_pred must contain integers",
        ),
        (
            np.array([0, -1]),
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            "y_true cannot contain negative labels",
        ),
        (
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([0.5, 0.5]),
            "y_pred cannot contain negative labels",
        ),
        (
            np.array([0, 1]),
            np.array([0]),
            np.array([0.5, 0.5]),
            "same number of samples",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([[0.5, 0.5]]),
            "sample_weights must be a one-dimensional array",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([1.0]),
            "sample_weights must contain the same number of samples",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.5, -0.5]),
            "sample_weights cannot contain negative values",
        ),
        (
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.2, 0.2]),
            "sample_weights must sum to 1.0",
        ),
    ],
)
def test_evaluate_weighted_predictions_rejects_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid weighted error inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        evaluate_weighted_predictions(
            y_true=y_true,
            y_pred=y_pred,
            sample_weights=sample_weights,
        )
