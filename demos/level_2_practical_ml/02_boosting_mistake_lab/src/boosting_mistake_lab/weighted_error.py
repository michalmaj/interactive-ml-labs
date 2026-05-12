"""Weighted error utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

type BoolArray = NDArray[np.bool_]
type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

EXPECTED_VECTOR_DIMENSIONS: Final[int] = 1
MIN_SAMPLE_COUNT: Final[int] = 1
NORMALIZED_WEIGHT_SUM: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class WeightedErrorResult:
    """Weighted prediction evaluation result.

    Attributes:
        weighted_error: Sum of weights assigned to misclassified samples.
        weighted_accuracy: Sum of weights assigned to correctly classified samples.
        mistake_mask: Boolean mask indicating misclassified samples.
        correct_mask: Boolean mask indicating correctly classified samples.
        mistake_weight_sum: Sum of weights assigned to mistakes.
        correct_weight_sum: Sum of weights assigned to correct predictions.
    """

    weighted_error: float
    weighted_accuracy: float
    mistake_mask: BoolArray
    correct_mask: BoolArray
    mistake_weight_sum: float
    correct_weight_sum: float


def evaluate_weighted_predictions(
    *,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weights: ArrayLike,
) -> WeightedErrorResult:
    """Evaluate predictions using normalized sample weights.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        sample_weights: Normalized per-sample weights.

    Returns:
        WeightedErrorResult with weighted error, weighted accuracy, and masks.

    Raises:
        ValueError: If labels or sample weights are invalid.
    """
    true_labels = _as_label_vector(y_true, name="y_true")
    predicted_labels = _as_label_vector(y_pred, name="y_pred")
    weights = _as_weight_vector(sample_weights)

    _validate_matching_lengths(
        y_true=true_labels,
        y_pred=predicted_labels,
        sample_weights=weights,
    )

    mistake_mask = true_labels != predicted_labels
    correct_mask = ~mistake_mask

    mistake_weight_sum = float(np.sum(weights[mistake_mask]))
    correct_weight_sum = float(np.sum(weights[correct_mask]))

    return WeightedErrorResult(
        weighted_error=mistake_weight_sum,
        weighted_accuracy=correct_weight_sum,
        mistake_mask=mistake_mask,
        correct_mask=correct_mask,
        mistake_weight_sum=mistake_weight_sum,
        correct_weight_sum=correct_weight_sum,
    )


def weighted_error_score(
    *,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weights: ArrayLike,
) -> float:
    """Compute weighted classification error.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        sample_weights: Normalized per-sample weights.

    Returns:
        Sum of weights assigned to misclassified samples.
    """
    return evaluate_weighted_predictions(
        y_true=y_true,
        y_pred=y_pred,
        sample_weights=sample_weights,
    ).weighted_error


def weighted_accuracy_score(
    *,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weights: ArrayLike,
) -> float:
    """Compute weighted classification accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        sample_weights: Normalized per-sample weights.

    Returns:
        Sum of weights assigned to correctly classified samples.
    """
    return evaluate_weighted_predictions(
        y_true=y_true,
        y_pred=y_pred,
        sample_weights=sample_weights,
    ).weighted_accuracy


def _as_label_vector(values: ArrayLike, *, name: str) -> IntArray:
    """Convert values to a one-dimensional integer label vector."""
    labels = np.asarray(values)

    if labels.ndim != EXPECTED_VECTOR_DIMENSIONS:
        msg = f"{name} must be a one-dimensional array."
        raise ValueError(msg)

    if labels.shape[0] < MIN_SAMPLE_COUNT:
        msg = f"{name} must contain at least one sample."
        raise ValueError(msg)

    if not np.issubdtype(labels.dtype, np.integer):
        msg = f"{name} must contain integers."
        raise ValueError(msg)

    integer_labels = labels.astype(int)

    if np.any(integer_labels < 0):
        msg = f"{name} cannot contain negative labels."
        raise ValueError(msg)

    return integer_labels


def _as_weight_vector(values: ArrayLike) -> FloatArray:
    """Convert values to a normalized one-dimensional weight vector."""
    weights = np.asarray(values, dtype=float)

    if weights.ndim != EXPECTED_VECTOR_DIMENSIONS:
        msg = "sample_weights must be a one-dimensional array."
        raise ValueError(msg)

    if weights.shape[0] < MIN_SAMPLE_COUNT:
        msg = "sample_weights must contain at least one sample."
        raise ValueError(msg)

    if np.any(weights < 0.0):
        msg = "sample_weights cannot contain negative values."
        raise ValueError(msg)

    if not np.isclose(float(np.sum(weights)), NORMALIZED_WEIGHT_SUM):
        msg = "sample_weights must sum to 1.0."
        raise ValueError(msg)

    return weights


def _validate_matching_lengths(
    *,
    y_true: IntArray,
    y_pred: IntArray,
    sample_weights: FloatArray,
) -> None:
    """Validate that labels and weights have matching lengths."""
    if y_true.shape[0] != y_pred.shape[0]:
        msg = (
            "y_true and y_pred must contain the same number of samples. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )
        raise ValueError(msg)

    if y_true.shape[0] != sample_weights.shape[0]:
        msg = (
            "sample_weights must contain the same number of samples as y_true. "
            f"Got {sample_weights.shape[0]} and {y_true.shape[0]}."
        )
        raise ValueError(msg)
