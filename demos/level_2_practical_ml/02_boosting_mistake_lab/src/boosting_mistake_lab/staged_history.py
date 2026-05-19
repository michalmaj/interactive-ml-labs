"""Staged boosted accuracy history for the Boosting Mistake Lab demo."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from boosting_mistake_lab.boosted_prediction import (
    boosted_accuracy_score,
    predict_boosted_ensemble,
)
from boosting_mistake_lab.boosting_round import BoostingRoundResult

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1
MIN_SAMPLE_COUNT: Final[int] = 1


@dataclass(frozen=True, slots=True)
class StagedBoostingHistory:
    """Boosted ensemble metrics computed after each completed round.

    Attributes:
        round_indices: One-based round indices.
        boosted_train_accuracies: Boosted train accuracy after each stage.
        boosted_test_accuracies: Boosted test accuracy after each stage.
        boosted_generalization_gaps: Train-test accuracy gaps after each stage.
        mean_train_confidences: Mean train confidence after each stage.
        mean_test_confidences: Mean test confidence after each stage.
        learner_weights: Learner alpha values.
        weighted_train_errors: Weighted train errors of individual weak learners.
    """

    round_indices: IntArray
    boosted_train_accuracies: FloatArray
    boosted_test_accuracies: FloatArray
    boosted_generalization_gaps: FloatArray
    mean_train_confidences: FloatArray
    mean_test_confidences: FloatArray
    learner_weights: FloatArray
    weighted_train_errors: FloatArray

    @property
    def final_train_accuracy(self) -> float:
        """Return final staged boosted train accuracy."""
        return float(self.boosted_train_accuracies[-1])

    @property
    def final_test_accuracy(self) -> float:
        """Return final staged boosted test accuracy."""
        return float(self.boosted_test_accuracies[-1])

    @property
    def final_generalization_gap(self) -> float:
        """Return final staged boosted generalization gap."""
        return float(self.boosted_generalization_gaps[-1])


def build_staged_boosting_history(
    *,
    round_results: Sequence[BoostingRoundResult],
    train_features: ArrayLike,
    train_targets: ArrayLike,
    test_features: ArrayLike,
    test_targets: ArrayLike,
) -> StagedBoostingHistory:
    """Build staged boosted train/test metrics.

    Args:
        round_results: Completed boosting round results.
        train_features: Training feature matrix.
        train_targets: Training labels.
        test_features: Test feature matrix.
        test_targets: Test labels.

    Returns:
        StagedBoostingHistory with one row per boosting stage.

    Raises:
        ValueError: If inputs are invalid.
    """
    rounds = _validate_round_results(round_results)
    train_feature_values = _as_feature_matrix(train_features, name="train_features")
    test_feature_values = _as_feature_matrix(test_features, name="test_features")
    train_target_values = _as_target_vector(train_targets, name="train_targets")
    test_target_values = _as_target_vector(test_targets, name="test_targets")

    _validate_feature_target_match(
        features=train_feature_values,
        targets=train_target_values,
        feature_name="train_features",
        target_name="train_targets",
    )
    _validate_feature_target_match(
        features=test_feature_values,
        targets=test_target_values,
        feature_name="test_features",
        target_name="test_targets",
    )

    learner_weights = np.asarray(
        [round_result.learner_weight for round_result in rounds],
        dtype=float,
    )
    weighted_train_errors = np.asarray(
        [round_result.weighted_train_error for round_result in rounds],
        dtype=float,
    )
    round_indices = np.asarray(
        [round_result.round_index for round_result in rounds],
        dtype=int,
    )

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []
    train_confidences: list[float] = []
    test_confidences: list[float] = []

    for stage_end in range(1, len(rounds) + 1):
        stage_rounds = rounds[:stage_end]
        stage_learners = tuple(round_result.weak_learner for round_result in stage_rounds)
        stage_weights = learner_weights[:stage_end]

        train_prediction = predict_boosted_ensemble(
            weak_learners=stage_learners,
            learner_weights=stage_weights,
            features=train_feature_values,
        )
        test_prediction = predict_boosted_ensemble(
            weak_learners=stage_learners,
            learner_weights=stage_weights,
            features=test_feature_values,
        )

        train_accuracies.append(
            boosted_accuracy_score(
                y_true=train_target_values,
                y_pred=train_prediction.predictions,
            ),
        )
        test_accuracies.append(
            boosted_accuracy_score(
                y_true=test_target_values,
                y_pred=test_prediction.predictions,
            ),
        )
        train_confidences.append(float(np.mean(train_prediction.confidence)))
        test_confidences.append(float(np.mean(test_prediction.confidence)))

    boosted_train_accuracies = np.asarray(train_accuracies, dtype=float)
    boosted_test_accuracies = np.asarray(test_accuracies, dtype=float)

    return StagedBoostingHistory(
        round_indices=round_indices,
        boosted_train_accuracies=boosted_train_accuracies,
        boosted_test_accuracies=boosted_test_accuracies,
        boosted_generalization_gaps=boosted_train_accuracies - boosted_test_accuracies,
        mean_train_confidences=np.asarray(train_confidences, dtype=float),
        mean_test_confidences=np.asarray(test_confidences, dtype=float),
        learner_weights=learner_weights,
        weighted_train_errors=weighted_train_errors,
    )


def _validate_round_results(
    round_results: Sequence[BoostingRoundResult],
) -> tuple[BoostingRoundResult, ...]:
    """Validate completed boosting rounds."""
    rounds = tuple(round_results)

    if not rounds:
        msg = "round_results must contain at least one boosting round."
        raise ValueError(msg)

    return rounds


def _as_feature_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    """Convert values to a two-dimensional feature matrix."""
    features = np.asarray(values, dtype=float)

    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = f"{name} must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[0] < MIN_SAMPLE_COUNT:
        msg = f"{name} must contain at least one sample."
        raise ValueError(msg)

    return features


def _as_target_vector(values: ArrayLike, *, name: str) -> IntArray:
    """Convert values to a one-dimensional integer target vector."""
    targets = np.asarray(values)

    if targets.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = f"{name} must be a one-dimensional array."
        raise ValueError(msg)

    if targets.shape[0] < MIN_SAMPLE_COUNT:
        msg = f"{name} must contain at least one sample."
        raise ValueError(msg)

    if not np.issubdtype(targets.dtype, np.integer):
        msg = f"{name} must contain integers."
        raise ValueError(msg)

    return targets.astype(int)


def _validate_feature_target_match(
    *,
    features: FloatArray,
    targets: IntArray,
    feature_name: str,
    target_name: str,
) -> None:
    """Validate feature/target sample count match."""
    if features.shape[0] != targets.shape[0]:
        msg = (
            f"{feature_name} and {target_name} must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)
