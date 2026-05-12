"""Sample weight update utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from boosting_mistake_lab.weighted_error import evaluate_weighted_predictions

type BoolArray = NDArray[np.bool_]
type FloatArray = NDArray[np.float64]

NORMALIZED_WEIGHT_SUM: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class SampleWeightUpdateResult:
    """Result of AdaBoost-style sample weight update.

    Attributes:
        old_weights: Previous normalized sample weights.
        updated_weights: New normalized sample weights.
        mistake_mask: Boolean mask for misclassified samples.
        correct_mask: Boolean mask for correctly classified samples.
        learner_weight: Learner contribution weight, commonly called alpha.
        mistake_multiplier: Multiplier applied to misclassified samples.
        correct_multiplier: Multiplier applied to correctly classified samples.
        normalization_factor: Sum of unnormalized updated weights.
        old_mistake_weight_sum: Previous total weight of mistakes.
        old_correct_weight_sum: Previous total weight of correct predictions.
        updated_mistake_weight_sum: New total weight of mistakes.
        updated_correct_weight_sum: New total weight of correct predictions.
    """

    old_weights: FloatArray
    updated_weights: FloatArray
    mistake_mask: BoolArray
    correct_mask: BoolArray
    learner_weight: float
    mistake_multiplier: float
    correct_multiplier: float
    normalization_factor: float
    old_mistake_weight_sum: float
    old_correct_weight_sum: float
    updated_mistake_weight_sum: float
    updated_correct_weight_sum: float


def update_sample_weights(
    *,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weights: ArrayLike,
    learner_weight: float,
) -> SampleWeightUpdateResult:
    """Update sample weights after one weak learner.

    The update follows the AdaBoost intuition:

    ```text
    correctly classified samples   -> multiplied by exp(-alpha)
    misclassified samples          -> multiplied by exp(+alpha)
    normalization                  -> sum(updated_weights) = 1.0
    ```

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        sample_weights: Current normalized sample weights.
        learner_weight: Learner contribution weight, commonly called alpha.

    Returns:
        SampleWeightUpdateResult with normalized updated weights and diagnostics.

    Raises:
        ValueError: If labels, sample weights, or learner weight are invalid.
    """
    _validate_learner_weight(learner_weight)

    weighted_result = evaluate_weighted_predictions(
        y_true=y_true,
        y_pred=y_pred,
        sample_weights=sample_weights,
    )
    old_weights = np.asarray(sample_weights, dtype=float)

    mistake_multiplier = exp(learner_weight)
    correct_multiplier = exp(-learner_weight)

    unnormalized_weights = np.where(
        weighted_result.mistake_mask,
        old_weights * mistake_multiplier,
        old_weights * correct_multiplier,
    )
    normalization_factor = float(np.sum(unnormalized_weights))

    if not isfinite(normalization_factor) or normalization_factor <= 0.0:
        msg = "normalization_factor must be finite and greater than 0."
        raise ValueError(msg)

    updated_weights = unnormalized_weights / normalization_factor

    return SampleWeightUpdateResult(
        old_weights=old_weights,
        updated_weights=updated_weights,
        mistake_mask=weighted_result.mistake_mask,
        correct_mask=weighted_result.correct_mask,
        learner_weight=learner_weight,
        mistake_multiplier=mistake_multiplier,
        correct_multiplier=correct_multiplier,
        normalization_factor=normalization_factor,
        old_mistake_weight_sum=weighted_result.mistake_weight_sum,
        old_correct_weight_sum=weighted_result.correct_weight_sum,
        updated_mistake_weight_sum=float(np.sum(updated_weights[weighted_result.mistake_mask])),
        updated_correct_weight_sum=float(np.sum(updated_weights[weighted_result.correct_mask])),
    )


def _validate_learner_weight(learner_weight: float) -> None:
    """Validate learner weight."""
    if not isfinite(learner_weight):
        msg = "learner_weight must be finite."
        raise ValueError(msg)
