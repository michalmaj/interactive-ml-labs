"""Weak learner baseline for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import ArrayLike, NDArray

from boosting_mistake_lab.dataset import WeightedTrainTestDataset
from boosting_mistake_lab.learner_weight import compute_learner_weight
from boosting_mistake_lab.weighted_error import evaluate_weighted_predictions

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type BoolArray = NDArray[np.bool_]

DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 1
MIN_SAMPLES_LEAF_LOWER_BOUND: Final[int] = 1
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class WeakLearnerConfig:
    """Configuration for a simple decision-stump weak learner.

    Attributes:
        min_samples_leaf: Minimum number of samples required in each child split.
    """

    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF


@dataclass(frozen=True, slots=True)
class WeakLearnerSplit:
    """One axis-aligned decision-stump split.

    Attributes:
        feature_index: Feature used for the split.
        threshold: Split threshold.
        left_prediction: Prediction for samples where feature <= threshold.
        right_prediction: Prediction for samples where feature > threshold.
        training_error: Unweighted training error of the split.
    """

    feature_index: int
    threshold: float
    left_prediction: int
    right_prediction: int
    training_error: float


class WeakLearnerBaseline:
    """A simple decision-stump weak learner baseline.

    The split is still selected using unweighted training error. Weighted error
    and learner weight are computed after fitting, which prepares the project
    for the next boosting step: sample weight update.
    """

    name: str = "weak_learner_baseline"

    def __init__(self, config: WeakLearnerConfig | None = None) -> None:
        """Initialize the weak learner."""
        self._config = config or WeakLearnerConfig()
        _validate_config(self._config)

        self._split: WeakLearnerSplit | None = None
        self._snapshot: AlgorithmSnapshot | None = None

    @property
    def split(self) -> WeakLearnerSplit:
        """Return the fitted split.

        Raises:
            RuntimeError: If the weak learner has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._split is not None

        return self._split

    def reset(self, dataset: WeightedTrainTestDataset) -> AlgorithmSnapshot:
        """Fit the weak learner and evaluate it on train/test splits.

        Args:
            dataset: Weighted train/test dataset.

        Returns:
            Algorithm snapshot with weak learner metrics.
        """
        train_features, train_targets = _extract_split_arrays(
            features=dataset.train.snapshot.features,
            targets=dataset.train.snapshot.targets,
            split_name="train",
        )
        test_features, test_targets = _extract_split_arrays(
            features=dataset.test.snapshot.features,
            targets=dataset.test.snapshot.targets,
            split_name="test",
        )

        _validate_sample_weights(
            sample_weights=dataset.train.sample_weights,
            sample_count=train_targets.shape[0],
            split_name="train",
        )
        _validate_sample_weights(
            sample_weights=dataset.test.sample_weights,
            sample_count=test_targets.shape[0],
            split_name="test",
        )

        split = _find_best_stump_split(
            features=train_features,
            targets=train_targets,
            min_samples_leaf=self._config.min_samples_leaf,
        )
        train_predictions = _predict_with_split(train_features, split)
        test_predictions = _predict_with_split(test_features, split)

        train_accuracy = _accuracy_score(train_targets, train_predictions)
        test_accuracy = _accuracy_score(test_targets, test_predictions)

        train_weighted_result = evaluate_weighted_predictions(
            y_true=train_targets,
            y_pred=train_predictions,
            sample_weights=dataset.train.sample_weights,
        )
        test_weighted_result = evaluate_weighted_predictions(
            y_true=test_targets,
            y_pred=test_predictions,
            sample_weights=dataset.test.sample_weights,
        )
        learner_weight_result = compute_learner_weight(
            weighted_error=train_weighted_result.weighted_error,
        )

        snapshot = AlgorithmSnapshot(
            iteration=1,
            status="fitted",
            metrics={
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_error": 1.0 - train_accuracy,
                "test_error": 1.0 - test_accuracy,
                "weighted_train_accuracy": train_weighted_result.weighted_accuracy,
                "weighted_train_error": train_weighted_result.weighted_error,
                "weighted_test_accuracy": test_weighted_result.weighted_accuracy,
                "weighted_test_error": test_weighted_result.weighted_error,
                "learner_weight": learner_weight_result.learner_weight,
                "clipped_weighted_train_error": (learner_weight_result.clipped_weighted_error),
                "feature_index": split.feature_index,
                "threshold": split.threshold,
                "left_prediction": split.left_prediction,
                "right_prediction": split.right_prediction,
                "min_samples_leaf": self._config.min_samples_leaf,
            },
            visual_state={
                "train_features": train_features,
                "train_targets": train_targets,
                "train_predictions": train_predictions,
                "train_mistakes": train_weighted_result.mistake_mask,
                "train_correct": train_weighted_result.correct_mask,
                "train_sample_weights": dataset.train.sample_weights,
                "test_features": test_features,
                "test_targets": test_targets,
                "test_predictions": test_predictions,
                "test_mistakes": test_weighted_result.mistake_mask,
                "test_correct": test_weighted_result.correct_mask,
                "test_sample_weights": dataset.test.sample_weights,
                "split": split,
                "learner_weight_result": learner_weight_result,
            },
            annotations=(
                "Weak learner baseline fitted as a decision stump.",
                f"Train accuracy: {train_accuracy:.3f}; test accuracy: {test_accuracy:.3f}.",
                f"Weighted train error: {train_weighted_result.weighted_error:.3f}.",
                f"Learner weight: {learner_weight_result.learner_weight:.3f}.",
            ),
            done=True,
        )

        self._split = split
        self._snapshot = snapshot

        return snapshot

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current weak learner snapshot.

        Raises:
            RuntimeError: If the weak learner has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._snapshot is not None

        return self._snapshot

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict labels using the fitted weak learner.

        Args:
            features: Feature matrix or a single feature vector.

        Returns:
            Predicted labels.

        Raises:
            RuntimeError: If the weak learner has not been fitted yet.
        """
        self._ensure_fitted()

        feature_values = _as_feature_matrix(features)

        assert self._split is not None

        return _predict_with_split(feature_values, self._split)

    def _ensure_fitted(self) -> None:
        """Ensure that the weak learner has been fitted."""
        if self._split is None or self._snapshot is None:
            msg = "The weak learner must be reset with a dataset before use."
            raise RuntimeError(msg)


def _find_best_stump_split(
    *,
    features: FloatArray,
    targets: IntArray,
    min_samples_leaf: int,
) -> WeakLearnerSplit:
    """Find the best unweighted decision-stump split."""
    best_split: WeakLearnerSplit | None = None

    for feature_index in range(features.shape[1]):
        thresholds = _candidate_thresholds(features[:, feature_index])

        for threshold in thresholds:
            candidate = _evaluate_candidate(
                features=features,
                targets=targets,
                feature_index=feature_index,
                threshold=float(threshold),
                min_samples_leaf=min_samples_leaf,
            )

            if candidate is None:
                continue

            if best_split is None or candidate.training_error < best_split.training_error:
                best_split = candidate

    if best_split is None:
        msg = "No valid weak learner split found."
        raise ValueError(msg)

    return best_split


def _evaluate_candidate(
    *,
    features: FloatArray,
    targets: IntArray,
    feature_index: int,
    threshold: float,
    min_samples_leaf: int,
) -> WeakLearnerSplit | None:
    """Evaluate one decision-stump candidate."""
    left_mask = features[:, feature_index] <= threshold
    right_mask = ~left_mask

    left_count = int(np.sum(left_mask))
    right_count = int(np.sum(right_mask))

    if left_count < min_samples_leaf or right_count < min_samples_leaf:
        return None

    left_prediction = _majority_class(targets[left_mask])
    right_prediction = _majority_class(targets[right_mask])

    predictions = np.empty_like(targets)
    predictions[left_mask] = left_prediction
    predictions[right_mask] = right_prediction

    training_error = float(np.mean(predictions != targets))

    return WeakLearnerSplit(
        feature_index=feature_index,
        threshold=threshold,
        left_prediction=left_prediction,
        right_prediction=right_prediction,
        training_error=training_error,
    )


def _candidate_thresholds(values: FloatArray) -> FloatArray:
    """Generate midpoint thresholds between sorted unique feature values."""
    unique_values = np.unique(values)

    if unique_values.shape[0] < 2:
        return np.array([], dtype=float)

    return (unique_values[:-1] + unique_values[1:]) / 2.0


def _majority_class(targets: IntArray) -> int:
    """Return majority class label with deterministic tie-breaking."""
    counts = np.bincount(targets)

    return int(np.argmax(counts))


def _predict_with_split(features: FloatArray, split: WeakLearnerSplit) -> IntArray:
    """Predict labels using one fitted split."""
    feature_values = _as_feature_matrix(features)
    predictions = np.empty(feature_values.shape[0], dtype=int)

    left_mask = feature_values[:, split.feature_index] <= split.threshold
    predictions[left_mask] = split.left_prediction
    predictions[~left_mask] = split.right_prediction

    return predictions


def _extract_split_arrays(
    *,
    features: object,
    targets: object,
    split_name: str,
) -> tuple[FloatArray, IntArray]:
    """Extract and validate one dataset split."""
    if targets is None:
        msg = f"{split_name} targets are required for weak learner evaluation."
        raise ValueError(msg)

    feature_values = _as_feature_matrix(features)
    target_values = np.asarray(targets)

    if target_values.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = f"{split_name} targets must be a one-dimensional array."
        raise ValueError(msg)

    if not np.issubdtype(target_values.dtype, np.integer):
        msg = f"{split_name} targets must contain integers."
        raise ValueError(msg)

    integer_targets = target_values.astype(int)

    if feature_values.shape[0] != integer_targets.shape[0]:
        msg = (
            f"{split_name} features and targets must contain the same number of samples. "
            f"Got {feature_values.shape[0]} and {integer_targets.shape[0]}."
        )
        raise ValueError(msg)

    if np.any(integer_targets < 0):
        msg = f"{split_name} targets cannot contain negative values."
        raise ValueError(msg)

    return feature_values, integer_targets


def _as_feature_matrix(features: object) -> FloatArray:
    """Convert features to a two-dimensional float matrix."""
    feature_values = np.asarray(features, dtype=float)

    if feature_values.ndim == 1:
        feature_values = feature_values.reshape(1, -1)

    if feature_values.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if feature_values.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    return feature_values


def _validate_sample_weights(
    *,
    sample_weights: FloatArray,
    sample_count: int,
    split_name: str,
) -> None:
    """Validate sample weights for future boosting steps."""
    weights = np.asarray(sample_weights, dtype=float)

    if weights.ndim != 1:
        msg = f"{split_name} sample weights must be a one-dimensional array."
        raise ValueError(msg)

    if weights.shape[0] != sample_count:
        msg = (
            f"{split_name} sample weights must match sample count. "
            f"Got {weights.shape[0]} and {sample_count}."
        )
        raise ValueError(msg)

    if np.any(weights < 0.0):
        msg = f"{split_name} sample weights cannot be negative."
        raise ValueError(msg)

    if not np.isclose(np.sum(weights), 1.0):
        msg = f"{split_name} sample weights must sum to 1.0."
        raise ValueError(msg)


def _accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    """Compute classification accuracy."""
    if y_true.shape[0] != y_pred.shape[0]:
        msg = (
            "y_true and y_pred must contain the same number of samples. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )
        raise ValueError(msg)

    return float(np.mean(y_true == y_pred))


def _validate_config(config: WeakLearnerConfig) -> None:
    """Validate weak learner configuration."""
    if config.min_samples_leaf < MIN_SAMPLES_LEAF_LOWER_BOUND:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)
