"""Boosted ensemble prediction utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from boosting_mistake_lab.weak_learner import WeakLearnerBaseline

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1
EXPECTED_VECTOR_DIMENSIONS: Final[int] = 1
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
MIN_SAMPLE_COUNT: Final[int] = 1


@dataclass(frozen=True, slots=True)
class BoostedPredictionResult:
    """Boosted ensemble prediction result.

    Attributes:
        predictions: Final binary ensemble predictions.
        confidence: Confidence derived from normalized absolute margin.
        raw_scores: Raw weighted signed vote scores.
        normalized_margins: Raw scores divided by sum of absolute learner weights.
        learner_predictions: Matrix of per-learner predictions.
        learner_weights: Learner contribution weights, commonly called alpha values.
    """

    predictions: IntArray
    confidence: FloatArray
    raw_scores: FloatArray
    normalized_margins: FloatArray
    learner_predictions: IntArray
    learner_weights: FloatArray


def predict_boosted_ensemble(
    *,
    weak_learners: Sequence[WeakLearnerBaseline],
    learner_weights: ArrayLike,
    features: ArrayLike,
) -> BoostedPredictionResult:
    """Predict labels using a weighted boosted ensemble.

    Binary class labels are converted internally:

    ```text
    class 0 -> -1
    class 1 -> +1
    ```

    The final score is:

    ```text
    score = sum(alpha_t * signed_prediction_t)
    ```

    Final prediction uses deterministic tie-breaking:

    ```text
    score > 0 -> class 1
    score <= 0 -> class 0
    ```

    Args:
        weak_learners: Fitted weak learners.
        learner_weights: Per-learner alpha values.
        features: Feature matrix or a single feature vector.

    Returns:
        BoostedPredictionResult with final predictions and confidence.

    Raises:
        ValueError: If learners, weights, or features are invalid.
        RuntimeError: If any weak learner has not been fitted yet.
    """
    learners = _validate_weak_learners(weak_learners)
    weights = _as_learner_weights(learner_weights, learner_count=len(learners))
    feature_values = _as_feature_matrix(features)

    learner_predictions = _collect_learner_predictions(
        weak_learners=learners,
        features=feature_values,
    )
    signed_predictions = _to_signed_predictions(learner_predictions)

    raw_scores = weights @ signed_predictions
    normalized_margins = _normalize_margins(
        raw_scores=raw_scores,
        learner_weights=weights,
    )
    predictions = np.where(raw_scores > 0.0, CLASS_ONE_LABEL, CLASS_ZERO_LABEL).astype(int)
    confidence = 0.5 + 0.5 * np.abs(normalized_margins)

    return BoostedPredictionResult(
        predictions=predictions,
        confidence=confidence,
        raw_scores=raw_scores,
        normalized_margins=normalized_margins,
        learner_predictions=learner_predictions,
        learner_weights=weights,
    )


def boosted_accuracy_score(*, y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute boosted ensemble classification accuracy.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.

    Returns:
        Accuracy score.

    Raises:
        ValueError: If inputs are invalid.
    """
    true_labels = _as_label_vector(y_true, name="y_true")
    predicted_labels = _as_label_vector(y_pred, name="y_pred")

    if true_labels.shape[0] != predicted_labels.shape[0]:
        msg = (
            "y_true and y_pred must contain the same number of samples. "
            f"Got {true_labels.shape[0]} and {predicted_labels.shape[0]}."
        )
        raise ValueError(msg)

    return float(np.mean(true_labels == predicted_labels))


def _collect_learner_predictions(
    *,
    weak_learners: tuple[WeakLearnerBaseline, ...],
    features: FloatArray,
) -> IntArray:
    """Collect per-learner predictions into a matrix."""
    predictions = [
        np.asarray(weak_learner.predict(features), dtype=int) for weak_learner in weak_learners
    ]

    prediction_matrix = np.vstack(predictions)
    _validate_prediction_matrix(prediction_matrix)

    return prediction_matrix


def _to_signed_predictions(predictions: IntArray) -> FloatArray:
    """Convert binary class predictions to signed AdaBoost votes."""
    return np.where(predictions == CLASS_ONE_LABEL, 1.0, -1.0)


def _normalize_margins(
    *,
    raw_scores: FloatArray,
    learner_weights: FloatArray,
) -> FloatArray:
    """Normalize raw ensemble scores to the range [-1, 1] when possible."""
    denominator = float(np.sum(np.abs(learner_weights)))

    if denominator == 0.0:
        return np.zeros_like(raw_scores, dtype=float)

    return np.clip(raw_scores / denominator, -1.0, 1.0)


def _validate_weak_learners(
    weak_learners: Sequence[WeakLearnerBaseline],
) -> tuple[WeakLearnerBaseline, ...]:
    """Validate weak learner sequence."""
    learners = tuple(weak_learners)

    if not learners:
        msg = "weak_learners must contain at least one fitted learner."
        raise ValueError(msg)

    return learners


def _as_learner_weights(values: ArrayLike, *, learner_count: int) -> FloatArray:
    """Convert learner weights to a one-dimensional float vector."""
    weights = np.asarray(values, dtype=float)

    if weights.ndim != EXPECTED_VECTOR_DIMENSIONS:
        msg = "learner_weights must be a one-dimensional array."
        raise ValueError(msg)

    if weights.shape[0] != learner_count:
        msg = (
            "learner_weights must match weak learner count. "
            f"Got {weights.shape[0]} and {learner_count}."
        )
        raise ValueError(msg)

    if np.any(~np.isfinite(weights)):
        msg = "learner_weights must contain finite values."
        raise ValueError(msg)

    return weights


def _as_feature_matrix(features: ArrayLike) -> FloatArray:
    """Convert features to a two-dimensional float matrix."""
    feature_values = np.asarray(features, dtype=float)

    if feature_values.ndim == 1:
        feature_values = feature_values.reshape(1, -1)

    if feature_values.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if feature_values.shape[0] < MIN_SAMPLE_COUNT:
        msg = "features must contain at least one sample."
        raise ValueError(msg)

    return feature_values


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

    if np.any(~np.isin(integer_labels, [CLASS_ZERO_LABEL, CLASS_ONE_LABEL])):
        msg = f"{name} must contain binary labels 0 or 1."
        raise ValueError(msg)

    return integer_labels


def _validate_prediction_matrix(predictions: IntArray) -> None:
    """Validate per-learner prediction matrix."""
    if predictions.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "learner predictions must form a two-dimensional matrix."
        raise ValueError(msg)

    if np.any(~np.isin(predictions, [CLASS_ZERO_LABEL, CLASS_ONE_LABEL])):
        msg = "weak learner predictions must contain binary labels 0 or 1."
        raise ValueError(msg)
