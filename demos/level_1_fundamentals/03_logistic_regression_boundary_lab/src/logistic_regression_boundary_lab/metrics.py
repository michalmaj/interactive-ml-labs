"""Metrics and probability helpers for Logistic Regression Boundary Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_THRESHOLD: Final[float] = 0.5
EPSILON: Final[float] = 1.0e-12

NEGATIVE_LABEL: Final[int] = 0
POSITIVE_LABEL: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ConfusionMatrixCounts:
    """Binary confusion matrix counts.

    Attributes:
        true_positive: Positive samples correctly predicted as positive.
        true_negative: Negative samples correctly predicted as negative.
        false_positive: Negative samples incorrectly predicted as positive.
        false_negative: Positive samples incorrectly predicted as negative.
    """

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Common binary classification metrics.

    Attributes:
        accuracy: Fraction of all correct predictions.
        precision: Fraction of predicted positives that are correct.
        recall: Fraction of actual positives that were found.
    """

    accuracy: float
    precision: float
    recall: float


def sigmoid(values: ArrayLike) -> FloatArray:
    """Apply the sigmoid function.

    The sigmoid function maps real-valued scores into the range `(0, 1)`,
    which makes it useful for interpreting logistic regression outputs as
    probabilities.

    Args:
        values: Scalar or array-like scores.

    Returns:
        NumPy array with sigmoid-transformed values.
    """
    scores = np.asarray(values, dtype=float)

    return 1.0 / (1.0 + np.exp(-scores))


def binary_cross_entropy(y_true: ArrayLike, y_probability: ArrayLike) -> float:
    """Compute binary cross-entropy loss.

    Args:
        y_true: Expected binary labels.
        y_probability: Predicted probabilities for the positive class.

    Returns:
        Binary cross-entropy loss.

    Raises:
        ValueError: If inputs have invalid shapes, labels, or probabilities.
    """
    true_labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_probability, dtype=float)

    _validate_binary_labels(true_labels, name="y_true")
    _validate_same_shape(true_labels, probabilities)
    _validate_probabilities(probabilities)

    clipped_probabilities = np.clip(probabilities, EPSILON, 1.0 - EPSILON)

    loss = -np.mean(
        true_labels * np.log(clipped_probabilities)
        + (1 - true_labels) * np.log(1.0 - clipped_probabilities),
    )

    return float(loss)


def predict_labels_from_probabilities(
    probabilities: ArrayLike,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> IntArray:
    """Convert positive-class probabilities into binary class labels.

    Args:
        probabilities: Predicted probabilities for the positive class.
        threshold: Decision threshold. Values greater than or equal to the
            threshold are assigned to class `1`.

    Returns:
        Binary predicted labels.

    Raises:
        ValueError: If probabilities or threshold are invalid.
    """
    probability_values = np.asarray(probabilities, dtype=float)

    _validate_probabilities(probability_values)
    _validate_threshold(threshold)

    return (probability_values >= threshold).astype(int)


def confusion_matrix_counts(y_true: ArrayLike, y_pred: ArrayLike) -> ConfusionMatrixCounts:
    """Compute binary confusion matrix counts.

    Args:
        y_true: Expected binary labels.
        y_pred: Predicted binary labels.

    Returns:
        ConfusionMatrixCounts object.

    Raises:
        ValueError: If inputs have invalid shapes or labels.
    """
    true_labels, predicted_labels = _validated_label_pair(y_true, y_pred)

    true_positive = int(
        np.sum((true_labels == POSITIVE_LABEL) & (predicted_labels == POSITIVE_LABEL)),
    )
    true_negative = int(
        np.sum((true_labels == NEGATIVE_LABEL) & (predicted_labels == NEGATIVE_LABEL)),
    )
    false_positive = int(
        np.sum((true_labels == NEGATIVE_LABEL) & (predicted_labels == POSITIVE_LABEL)),
    )
    false_negative = int(
        np.sum((true_labels == POSITIVE_LABEL) & (predicted_labels == NEGATIVE_LABEL)),
    )

    return ConfusionMatrixCounts(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute binary classification accuracy."""
    true_labels, predicted_labels = _validated_label_pair(y_true, y_pred)

    return float(np.mean(true_labels == predicted_labels))


def precision_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute precision for the positive class.

    If there are no predicted positives, precision is defined as `0.0`.
    """
    counts = confusion_matrix_counts(y_true, y_pred)
    denominator = counts.true_positive + counts.false_positive

    if denominator == 0:
        return 0.0

    return counts.true_positive / denominator


def recall_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute recall for the positive class.

    If there are no actual positives, recall is defined as `0.0`.
    """
    counts = confusion_matrix_counts(y_true, y_pred)
    denominator = counts.true_positive + counts.false_negative

    if denominator == 0:
        return 0.0

    return counts.true_positive / denominator


def classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> ClassificationMetrics:
    """Compute common binary classification metrics."""
    return ClassificationMetrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
    )


def _validated_label_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[IntArray, IntArray]:
    """Convert and validate a pair of binary label arrays."""
    true_labels = np.asarray(y_true, dtype=int)
    predicted_labels = np.asarray(y_pred, dtype=int)

    _validate_binary_labels(true_labels, name="y_true")
    _validate_binary_labels(predicted_labels, name="y_pred")
    _validate_same_shape(true_labels, predicted_labels)

    return true_labels, predicted_labels


def _validate_binary_labels(labels: IntArray, *, name: str) -> None:
    """Validate binary label array."""
    if labels.ndim != 1:
        msg = f"{name} must be a one-dimensional array."
        raise ValueError(msg)

    if labels.size == 0:
        msg = f"{name} cannot be empty."
        raise ValueError(msg)

    unique_labels = set(labels.tolist())

    if not unique_labels.issubset({NEGATIVE_LABEL, POSITIVE_LABEL}):
        msg = f"{name} must contain only binary labels 0 and 1."
        raise ValueError(msg)


def _validate_probabilities(probabilities: FloatArray) -> None:
    """Validate probability array."""
    if probabilities.ndim == 0:
        return

    if probabilities.size == 0:
        msg = "probabilities cannot be empty."
        raise ValueError(msg)

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        msg = "probabilities must be in the range [0, 1]."
        raise ValueError(msg)


def _validate_threshold(threshold: float) -> None:
    """Validate decision threshold."""
    if not 0.0 <= threshold <= 1.0:
        msg = "threshold must be in the range [0, 1]."
        raise ValueError(msg)


def _validate_same_shape(first: ArrayLike, second: ArrayLike) -> None:
    """Validate that two arrays have the same shape."""
    first_array = np.asarray(first)
    second_array = np.asarray(second)

    if first_array.shape != second_array.shape:
        msg = f"Inputs must have the same shape. Got {first_array.shape} and {second_array.shape}."
        raise ValueError(msg)
