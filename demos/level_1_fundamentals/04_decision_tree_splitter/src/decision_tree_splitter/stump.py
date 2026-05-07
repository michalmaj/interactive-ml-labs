"""Decision stump model for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from numpy.typing import ArrayLike, NDArray

from decision_tree_splitter.split import (
    CRITERION_GINI,
    ImpurityCriterion,
    SplitEvaluation,
    best_split,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type BoolArray = NDArray[np.bool_]

DEFAULT_CRITERION: Final[ImpurityCriterion] = CRITERION_GINI
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_LABEL_DIMENSIONS: Final[int] = 1
MIN_FEATURE_COUNT: Final[int] = 1
MIN_SAMPLE_COUNT: Final[int] = 2


@dataclass(frozen=True, slots=True)
class DecisionStumpConfig:
    """Configuration for a decision stump.

    Attributes:
        criterion: Impurity criterion used to choose the best split.
    """

    criterion: ImpurityCriterion = DEFAULT_CRITERION


@dataclass(frozen=True, slots=True)
class LeafPrediction:
    """Prediction stored in one leaf of a decision stump.

    Attributes:
        class_label: Majority class predicted by the leaf.
        sample_count: Number of training samples assigned to the leaf.
        class_counts: Counts of training samples per class.
    """

    class_label: int
    sample_count: int
    class_counts: tuple[int, ...]


class DecisionStump:
    """One-level decision tree classifier.

    A decision stump contains exactly one root split and two leaves.

    The model is intentionally simple and useful for teaching the first step
    of decision-tree learning before introducing recursive trees.
    """

    name: str = "decision_stump"

    def __init__(self, config: DecisionStumpConfig | None = None) -> None:
        """Initialize the stump with optional configuration."""
        self._config = config or DecisionStumpConfig()

        self._features: FloatArray | None = None
        self._targets: IntArray | None = None
        self._split_evaluation: SplitEvaluation | None = None
        self._left_leaf: LeafPrediction | None = None
        self._right_leaf: LeafPrediction | None = None
        self._class_count: int | None = None
        self._iteration = 0

    @property
    def split_evaluation(self) -> SplitEvaluation:
        """Return the fitted root split evaluation.

        Raises:
            RuntimeError: If the stump has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._split_evaluation is not None

        return self._split_evaluation

    @property
    def left_leaf(self) -> LeafPrediction:
        """Return the left leaf prediction.

        Raises:
            RuntimeError: If the stump has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._left_leaf is not None

        return self._left_leaf

    @property
    def right_leaf(self) -> LeafPrediction:
        """Return the right leaf prediction.

        Raises:
            RuntimeError: If the stump has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._right_leaf is not None

        return self._right_leaf

    def reset(self, dataset: DatasetSnapshot) -> AlgorithmSnapshot:
        """Fit the decision stump from a dataset snapshot.

        Args:
            dataset: Dataset containing features and class labels.

        Returns:
            Snapshot of the fitted stump.

        Raises:
            ValueError: If the dataset is invalid or no split can be found.
        """
        features, targets = _extract_arrays(dataset)

        self._features = features
        self._targets = targets
        self._class_count = int(np.max(targets)) + 1
        self._split_evaluation = best_split(
            features,
            targets,
            criterion=self._config.criterion,
        )

        candidate = self._split_evaluation.candidate
        left_mask = features[:, candidate.feature_index] <= candidate.threshold
        right_mask = ~left_mask

        self._left_leaf = _make_leaf_prediction(
            targets[left_mask],
            class_count=self._class_count,
        )
        self._right_leaf = _make_leaf_prediction(
            targets[right_mask],
            class_count=self._class_count,
        )
        self._iteration = 1

        return self.snapshot()

    def step(self) -> AlgorithmSnapshot:
        """Return current snapshot.

        A stump is fitted in a single operation, so there is no iterative
        training process. The method exists to keep a familiar demo interface.
        """
        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current fitted stump state."""
        self._ensure_fitted()

        assert self._features is not None
        assert self._targets is not None
        assert self._split_evaluation is not None
        assert self._left_leaf is not None
        assert self._right_leaf is not None

        candidate = self._split_evaluation.candidate
        left_mask = self._features[:, candidate.feature_index] <= candidate.threshold
        right_mask = ~left_mask
        predictions = self.predict(self._features)
        training_accuracy = _accuracy_score(self._targets, predictions)

        return AlgorithmSnapshot(
            iteration=self._iteration,
            status="fitted",
            visual_state={
                "features": self._features,
                "targets": self._targets,
                "predictions": predictions,
                "left_mask": left_mask,
                "right_mask": right_mask,
                "split_evaluation": self._split_evaluation,
                "left_leaf": self._left_leaf,
                "right_leaf": self._right_leaf,
            },
            metrics={
                "training_accuracy": training_accuracy,
                "criterion": self._split_evaluation.criterion,
                "feature_index": candidate.feature_index,
                "threshold": candidate.threshold,
                "parent_impurity": self._split_evaluation.parent_impurity,
                "left_impurity": self._split_evaluation.left_impurity,
                "right_impurity": self._split_evaluation.right_impurity,
                "weighted_child_impurity": (self._split_evaluation.weighted_child_impurity),
                "information_gain": self._split_evaluation.information_gain,
                "left_sample_count": self._split_evaluation.left_sample_count,
                "right_sample_count": self._split_evaluation.right_sample_count,
                "left_prediction": self._left_leaf.class_label,
                "right_prediction": self._right_leaf.class_label,
            },
            annotations=(
                _build_split_annotation(self._split_evaluation),
                _build_leaf_annotation(
                    left_leaf=self._left_leaf,
                    right_leaf=self._right_leaf,
                ),
            ),
            done=True,
        )

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict class labels for input features.

        Args:
            features: Feature matrix or one feature vector.

        Returns:
            Predicted class labels.

        Raises:
            RuntimeError: If the stump has not been fitted yet.
            ValueError: If features are invalid.
        """
        self._ensure_fitted()

        assert self._features is not None
        assert self._split_evaluation is not None
        assert self._left_leaf is not None
        assert self._right_leaf is not None

        values = _prepare_prediction_features(
            features,
            expected_feature_count=self._features.shape[1],
        )
        candidate = self._split_evaluation.candidate
        left_mask = values[:, candidate.feature_index] <= candidate.threshold

        return np.where(
            left_mask,
            self._left_leaf.class_label,
            self._right_leaf.class_label,
        ).astype(int)

    def _ensure_fitted(self) -> None:
        """Ensure that the stump has been fitted."""
        if (
            self._features is None
            or self._targets is None
            or self._split_evaluation is None
            or self._left_leaf is None
            or self._right_leaf is None
        ):
            msg = "The decision stump must be reset with a dataset before use."
            raise RuntimeError(msg)


def _extract_arrays(dataset: DatasetSnapshot) -> tuple[FloatArray, IntArray]:
    """Extract and validate features and labels from a dataset."""
    if dataset.targets is None:
        msg = "Dataset targets are required for decision stump training."
        raise ValueError(msg)

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets)

    _validate_features(features)
    _validate_targets(targets)

    integer_targets = targets.astype(int)

    if features.shape[0] != integer_targets.shape[0]:
        msg = (
            "features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {integer_targets.shape[0]}."
        )
        raise ValueError(msg)

    if features.shape[0] < MIN_SAMPLE_COUNT:
        msg = "At least two samples are required to train a decision stump."
        raise ValueError(msg)

    return features, integer_targets


def _validate_features(features: FloatArray) -> None:
    """Validate feature matrix."""
    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    if features.shape[1] < MIN_FEATURE_COUNT:
        msg = "features must contain at least one column."
        raise ValueError(msg)


def _validate_targets(targets: NDArray[np.generic]) -> None:
    """Validate target labels."""
    if targets.ndim != EXPECTED_LABEL_DIMENSIONS:
        msg = "targets must be a one-dimensional array."
        raise ValueError(msg)

    if targets.size == 0:
        msg = "targets cannot be empty."
        raise ValueError(msg)

    if not np.issubdtype(targets.dtype, np.integer):
        msg = "targets must contain integers."
        raise ValueError(msg)

    if np.any(targets.astype(int) < 0):
        msg = "targets cannot contain negative values."
        raise ValueError(msg)


def _prepare_prediction_features(
    features: ArrayLike,
    *,
    expected_feature_count: int,
) -> FloatArray:
    """Convert prediction input into a valid two-dimensional matrix."""
    values = np.asarray(features, dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    _validate_features(values)

    if values.shape[1] != expected_feature_count:
        msg = (
            "features must contain the same number of columns as training data. "
            f"Got {values.shape[1]} and expected {expected_feature_count}."
        )
        raise ValueError(msg)

    return values


def _make_leaf_prediction(labels: IntArray, *, class_count: int) -> LeafPrediction:
    """Create a majority-class prediction for one leaf."""
    counts = np.bincount(labels, minlength=class_count)
    class_label = int(np.argmax(counts))

    return LeafPrediction(
        class_label=class_label,
        sample_count=int(labels.shape[0]),
        class_counts=tuple(int(value) for value in counts.tolist()),
    )


def _accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def _build_split_annotation(evaluation: SplitEvaluation) -> str:
    """Build a short split annotation."""
    candidate = evaluation.candidate

    return (
        f"Decision stump split: x{candidate.feature_index + 1} <= "
        f"{candidate.threshold:.3f}; gain={evaluation.information_gain:.4f}."
    )


def _build_leaf_annotation(
    *,
    left_leaf: LeafPrediction,
    right_leaf: LeafPrediction,
) -> str:
    """Build a short leaf prediction annotation."""
    return (
        f"Left leaf predicts class_{left_leaf.class_label}; "
        f"right leaf predicts class_{right_leaf.class_label}."
    )
