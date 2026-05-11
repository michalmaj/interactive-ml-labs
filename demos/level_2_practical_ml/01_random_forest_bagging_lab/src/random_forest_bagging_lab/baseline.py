"""Single-tree baseline for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from decision_tree_splitter import (
    DecisionTreeConfig,
    DecisionTreeNode,
    RecursiveDecisionTree,
)
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import ArrayLike, NDArray

from random_forest_bagging_lab.dataset import TrainTestDataset

type IntArray = NDArray[np.int_]

DEFAULT_MAX_DEPTH: Final[int] = 2
DEFAULT_CRITERION: Final[str] = "gini"
MIN_MAX_DEPTH: Final[int] = 1


@dataclass(frozen=True, slots=True)
class SingleTreeBaselineConfig:
    """Configuration for a single decision-tree baseline.

    Attributes:
        max_depth: Maximum depth of the baseline decision tree.
        criterion: Impurity criterion used by the decision tree.
    """

    max_depth: int = DEFAULT_MAX_DEPTH
    criterion: str = DEFAULT_CRITERION


class SingleTreeBaseline:
    """Train and evaluate one recursive decision tree on train/test data.

    This class is intentionally simple. It gives Random Forest Bagging Lab a
    baseline model that can later be compared with an ensemble of many trees.
    """

    name: str = "single_tree_baseline"

    def __init__(self, config: SingleTreeBaselineConfig | None = None) -> None:
        """Initialize the baseline model."""
        self._config = config or SingleTreeBaselineConfig()
        _validate_config(self._config)

        self._tree: RecursiveDecisionTree | None = None
        self._train_snapshot: AlgorithmSnapshot | None = None
        self._test_predictions: IntArray | None = None
        self._test_targets: IntArray | None = None
        self._snapshot: AlgorithmSnapshot | None = None

    @property
    def tree(self) -> RecursiveDecisionTree:
        """Return the fitted decision tree.

        Raises:
            RuntimeError: If the baseline has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._tree is not None

        return self._tree

    def reset(self, dataset: TrainTestDataset) -> AlgorithmSnapshot:
        """Fit the baseline tree and evaluate it on train and test splits.

        Args:
            dataset: Train/test dataset.

        Returns:
            Algorithm snapshot containing baseline metrics.

        Raises:
            ValueError: If the underlying tree cannot be trained or evaluated.
        """
        tree = RecursiveDecisionTree(
            DecisionTreeConfig(
                criterion=self._config.criterion,
                max_depth=self._config.max_depth,
            ),
        )

        train_snapshot = tree.reset(dataset.train)
        test_features = np.asarray(dataset.test.features, dtype=float)
        test_targets = _extract_test_targets(dataset)
        test_predictions = tree.predict(test_features)

        train_accuracy = float(train_snapshot.metrics["training_accuracy"])
        test_accuracy = _accuracy_score(test_targets, test_predictions)

        root = train_snapshot.visual_state["root"]

        if not isinstance(root, DecisionTreeNode):
            msg = "train snapshot root must be a DecisionTreeNode."
            raise TypeError(msg)

        snapshot = AlgorithmSnapshot(
            iteration=int(train_snapshot.metrics["node_count"]),
            status="fitted",
            visual_state={
                "train_features": dataset.train.features,
                "train_targets": dataset.train.targets,
                "test_features": dataset.test.features,
                "test_targets": test_targets,
                "test_predictions": test_predictions,
                "tree_root": root,
                "tree_nodes": train_snapshot.visual_state["nodes"],
            },
            metrics={
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "criterion": self._config.criterion,
                "max_depth": self._config.max_depth,
                "actual_depth": train_snapshot.metrics["actual_depth"],
                "node_count": train_snapshot.metrics["node_count"],
                "leaf_count": train_snapshot.metrics["leaf_count"],
            },
            annotations=(
                "Single-tree baseline fitted on the training split.",
                f"Train accuracy: {train_accuracy:.3f}; test accuracy: {test_accuracy:.3f}.",
            ),
            done=True,
        )

        self._tree = tree
        self._train_snapshot = train_snapshot
        self._test_predictions = test_predictions
        self._test_targets = test_targets
        self._snapshot = snapshot

        return snapshot

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current baseline snapshot.

        Raises:
            RuntimeError: If the baseline has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._snapshot is not None

        return self._snapshot

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict labels using the fitted baseline tree.

        Args:
            features: Feature matrix or one feature vector.

        Returns:
            Predicted labels.

        Raises:
            RuntimeError: If the baseline has not been fitted yet.
        """
        return self.tree.predict(features)

    def _ensure_fitted(self) -> None:
        """Ensure that the baseline has been fitted."""
        if self._tree is None or self._snapshot is None:
            msg = "The single-tree baseline must be reset with a dataset before use."
            raise RuntimeError(msg)


def _extract_test_targets(dataset: TrainTestDataset) -> IntArray:
    """Extract test targets from a train/test dataset."""
    if dataset.test.targets is None:
        msg = "Test targets are required for baseline evaluation."
        raise ValueError(msg)

    targets = np.asarray(dataset.test.targets)

    if targets.ndim != 1:
        msg = "Test targets must be a one-dimensional array."
        raise ValueError(msg)

    if not np.issubdtype(targets.dtype, np.integer):
        msg = "Test targets must contain integers."
        raise ValueError(msg)

    return targets.astype(int)


def _accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    """Compute classification accuracy."""
    if y_true.shape[0] != y_pred.shape[0]:
        msg = (
            "y_true and y_pred must contain the same number of samples. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )
        raise ValueError(msg)

    return float(np.mean(y_true == y_pred))


def _validate_config(config: SingleTreeBaselineConfig) -> None:
    """Validate baseline configuration."""
    if config.max_depth < MIN_MAX_DEPTH:
        msg = "max_depth must be greater than or equal to 1."
        raise ValueError(msg)

    if config.criterion not in {"gini", "entropy"}:
        msg = "criterion must be one of: entropy, gini."
        raise ValueError(msg)
