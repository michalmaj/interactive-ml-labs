"""Tests for logistic regression metrics and probability helpers."""

import numpy as np
import pytest
from logistic_regression_boundary_lab import (
    ClassificationMetrics,
    ConfusionMatrixCounts,
    accuracy_score,
    binary_cross_entropy,
    classification_metrics,
    confusion_matrix_counts,
    precision_score,
    predict_labels_from_probabilities,
    recall_score,
    sigmoid,
)

EXPECTED_SIGMOID_ZERO: float = 0.5
EXPECTED_ACCURACY: float = 0.75
EXPECTED_PRECISION: float = 2.0 / 3.0
EXPECTED_RECALL: float = 1.0


def test_sigmoid_maps_zero_to_one_half() -> None:
    """Sigmoid of zero should be exactly 0.5."""
    result = sigmoid([0.0])

    np.testing.assert_allclose(result, np.array([EXPECTED_SIGMOID_ZERO]))


def test_sigmoid_maps_values_to_probability_range() -> None:
    """Sigmoid values should be in the open interval (0, 1)."""
    result = sigmoid(np.array([-5.0, 0.0, 5.0]))

    assert np.all(result > 0.0)
    assert np.all(result < 1.0)
    assert result[0] < result[1] < result[2]


def test_predict_labels_from_probabilities_uses_threshold() -> None:
    """Probabilities should be converted to labels using the threshold."""
    probabilities = np.array([0.2, 0.5, 0.8])

    result = predict_labels_from_probabilities(probabilities, threshold=0.5)

    np.testing.assert_array_equal(result, np.array([0, 1, 1]))


def test_predict_labels_from_probabilities_accepts_custom_threshold() -> None:
    """Custom threshold should change predicted labels."""
    probabilities = np.array([0.2, 0.5, 0.8])

    result = predict_labels_from_probabilities(probabilities, threshold=0.7)

    np.testing.assert_array_equal(result, np.array([0, 0, 1]))


def test_binary_cross_entropy_is_low_for_good_predictions() -> None:
    """Good probability predictions should produce low BCE."""
    y_true = np.array([0, 1])
    probabilities = np.array([0.01, 0.99])

    result = binary_cross_entropy(y_true, probabilities)

    assert result < 0.02


def test_binary_cross_entropy_is_higher_for_bad_predictions() -> None:
    """Bad probability predictions should produce higher BCE."""
    y_true = np.array([0, 1])
    good_probabilities = np.array([0.01, 0.99])
    bad_probabilities = np.array([0.99, 0.01])

    good_loss = binary_cross_entropy(y_true, good_probabilities)
    bad_loss = binary_cross_entropy(y_true, bad_probabilities)

    assert bad_loss > good_loss


def test_confusion_matrix_counts_computes_binary_counts() -> None:
    """Confusion matrix counts should match expected binary outcomes."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = confusion_matrix_counts(y_true, y_pred)

    assert result == ConfusionMatrixCounts(
        true_positive=2,
        true_negative=1,
        false_positive=1,
        false_negative=0,
    )


def test_accuracy_score_computes_fraction_of_correct_predictions() -> None:
    """Accuracy should be the fraction of all correct predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = accuracy_score(y_true, y_pred)

    assert result == pytest.approx(EXPECTED_ACCURACY)


def test_precision_score_computes_positive_precision() -> None:
    """Precision should be TP / (TP + FP)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = precision_score(y_true, y_pred)

    assert result == pytest.approx(EXPECTED_PRECISION)


def test_recall_score_computes_positive_recall() -> None:
    """Recall should be TP / (TP + FN)."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = recall_score(y_true, y_pred)

    assert result == pytest.approx(EXPECTED_RECALL)


def test_classification_metrics_groups_common_metrics() -> None:
    """Classification metrics helper should return a grouped metrics object."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    result = classification_metrics(y_true, y_pred)

    assert result == ClassificationMetrics(
        accuracy=pytest.approx(EXPECTED_ACCURACY),
        precision=pytest.approx(EXPECTED_PRECISION),
        recall=pytest.approx(EXPECTED_RECALL),
    )


def test_precision_is_zero_without_predicted_positives() -> None:
    """Precision should be zero when there are no predicted positives."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 0])

    result = precision_score(y_true, y_pred)

    assert result == 0.0


def test_recall_is_zero_without_actual_positives() -> None:
    """Recall should be zero when there are no actual positives."""
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])

    result = recall_score(y_true, y_pred)

    assert result == 0.0


@pytest.mark.parametrize(
    "probabilities, expected_message",
    [
        (
            np.array([]),
            "probabilities cannot be empty",
        ),
        (
            np.array([-0.1, 0.5]),
            "range",
        ),
        (
            np.array([0.5, 1.1]),
            "range",
        ),
    ],
)
def test_probability_helpers_reject_invalid_probabilities(
    probabilities: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid probabilities should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        predict_labels_from_probabilities(probabilities)


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_predict_labels_rejects_invalid_threshold(threshold: float) -> None:
    """Threshold should be in the range [0, 1]."""
    with pytest.raises(ValueError, match="threshold"):
        predict_labels_from_probabilities([0.5], threshold=threshold)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_message",
    [
        (
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            "one-dimensional",
        ),
        (
            np.array([]),
            np.array([]),
            "cannot be empty",
        ),
        (
            np.array([0, 2]),
            np.array([0, 1]),
            "binary labels",
        ),
        (
            np.array([0, 1]),
            np.array([0]),
            "same shape",
        ),
    ],
)
def test_label_metrics_reject_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid labels should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        accuracy_score(y_true, y_pred)


def test_binary_cross_entropy_rejects_mismatched_shapes() -> None:
    """BCE inputs should have the same shape."""
    with pytest.raises(ValueError, match="same shape"):
        binary_cross_entropy([0, 1], [0.2])


def test_binary_cross_entropy_rejects_invalid_labels() -> None:
    """BCE should require binary labels."""
    with pytest.raises(ValueError, match="binary labels"):
        binary_cross_entropy([0, 2], [0.2, 0.8])


def test_binary_cross_entropy_rejects_invalid_probabilities() -> None:
    """BCE should require probabilities from [0, 1]."""
    with pytest.raises(ValueError, match="range"):
        binary_cross_entropy([0, 1], [0.2, 1.2])
