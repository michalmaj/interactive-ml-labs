"""Recursive decision tree model for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from numpy.typing import ArrayLike, NDArray

from decision_tree_splitter.impurity import entropy_impurity, gini_impurity
from decision_tree_splitter.split import (
    CRITERION_GINI,
    ImpurityCriterion,
    SplitEvaluation,
    evaluate_split,
    generate_split_candidates,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_CRITERION: Final[ImpurityCriterion] = CRITERION_GINI
DEFAULT_MAX_DEPTH: Final[int] = 2
DEFAULT_MIN_SAMPLES_SPLIT: Final[int] = 2
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 1
DEFAULT_MIN_INFORMATION_GAIN: Final[float] = 0.0

EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1
MIN_FEATURE_COUNT: Final[int] = 1
MIN_MAX_DEPTH: Final[int] = 1
MIN_SAMPLES_SPLIT_LOWER_BOUND: Final[int] = 2
MIN_SAMPLES_LEAF_LOWER_BOUND: Final[int] = 1
ROOT_DEPTH: Final[int] = 0


@dataclass(frozen=True, slots=True)
class DecisionTreeConfig:
    """Configuration for a recursive decision tree classifier.

    Attributes:
        criterion: Impurity criterion used for split scoring.
        max_depth: Maximum depth of the tree. Root has depth 0.
        min_samples_split: Minimum number of samples required to split a node.
        min_samples_leaf: Minimum number of samples required in each child leaf.
        min_information_gain: Minimum information gain required to accept a split.
            The default value is zero so the tree can demonstrate XOR, where the
            first useful split may have zero immediate gain.
    """

    criterion: ImpurityCriterion = DEFAULT_CRITERION
    max_depth: int = DEFAULT_MAX_DEPTH
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF
    min_information_gain: float = DEFAULT_MIN_INFORMATION_GAIN


@dataclass(frozen=True, slots=True)
class DecisionTreeNode:
    """One node of a recursive decision tree.

    Attributes:
        node_id: Stable node identifier.
        depth: Node depth. Root has depth 0.
        prediction: Majority-class prediction at this node.
        sample_count: Number of training samples assigned to the node.
        class_counts: Class counts inside the node.
        impurity: Node impurity before splitting.
        split_evaluation: Split evaluation if the node is internal.
        left: Left child node.
        right: Right child node.
    """

    node_id: int
    depth: int
    prediction: int
    sample_count: int
    class_counts: tuple[int, ...]
    impurity: float
    split_evaluation: SplitEvaluation | None = None
    left: DecisionTreeNode | None = None
    right: DecisionTreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        """Return whether the node is a leaf."""
        return self.split_evaluation is None or self.left is None or self.right is None


class RecursiveDecisionTree:
    """Small recursive decision tree classifier for educational demos."""

    name: str = "recursive_decision_tree"

    def __init__(self, config: DecisionTreeConfig | None = None) -> None:
        """Initialize the model with optional configuration."""
        self._config = config or DecisionTreeConfig()
        _validate_config(self._config)

        self._features: FloatArray | None = None
        self._targets: IntArray | None = None
        self._root: DecisionTreeNode | None = None
        self._class_count: int | None = None
        self._next_node_id = 0

    @property
    def root(self) -> DecisionTreeNode:
        """Return the fitted root node.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._root is not None

        return self._root

    def reset(self, dataset: DatasetSnapshot) -> AlgorithmSnapshot:
        """Fit the recursive decision tree from a dataset snapshot.

        Args:
            dataset: Dataset containing features and class labels.

        Returns:
            Snapshot of the fitted tree.

        Raises:
            ValueError: If the dataset is invalid or the tree cannot be fitted.
        """
        features, targets = _extract_arrays(dataset)

        self._features = features
        self._targets = targets
        self._class_count = int(np.max(targets)) + 1
        self._next_node_id = 0
        self._root = self._build_node(features, targets, depth=ROOT_DEPTH)

        return self.snapshot()

    def step(self) -> AlgorithmSnapshot:
        """Return current snapshot.

        The full tree is fitted in one operation. The method exists to keep a
        familiar demo interface shared with other labs.
        """
        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        """Return a snapshot of the fitted tree."""
        self._ensure_fitted()

        assert self._features is not None
        assert self._targets is not None
        assert self._root is not None

        predictions = self.predict(self._features)
        training_accuracy = _accuracy_score(self._targets, predictions)
        node_count = _count_nodes(self._root)
        leaf_count = _count_leaves(self._root)
        actual_depth = _max_depth(self._root)

        return AlgorithmSnapshot(
            iteration=node_count,
            status="fitted",
            visual_state={
                "features": self._features,
                "targets": self._targets,
                "predictions": predictions,
                "root": self._root,
                "nodes": tuple(_iter_nodes(self._root)),
            },
            metrics={
                "training_accuracy": training_accuracy,
                "criterion": self._config.criterion,
                "max_depth": self._config.max_depth,
                "actual_depth": actual_depth,
                "node_count": node_count,
                "leaf_count": leaf_count,
                "root_impurity": self._root.impurity,
                "root_prediction": self._root.prediction,
            },
            annotations=(
                f"Recursive tree fitted with max_depth={self._config.max_depth}.",
                f"Tree has {node_count} nodes and {leaf_count} leaves.",
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
            RuntimeError: If the tree has not been fitted yet.
            ValueError: If features are invalid.
        """
        self._ensure_fitted()

        assert self._features is not None
        assert self._root is not None

        values = _prepare_prediction_features(
            features,
            expected_feature_count=self._features.shape[1],
        )

        return np.array(
            [self._predict_one(row, self._root) for row in values],
            dtype=int,
        )

    def _build_node(
        self,
        features: FloatArray,
        targets: IntArray,
        *,
        depth: int,
    ) -> DecisionTreeNode:
        """Build one tree node recursively."""
        node_id = self._allocate_node_id()
        prediction = _majority_class(targets)
        class_counts = _class_counts(targets, class_count=self._class_count_value())
        impurity = _compute_impurity(
            targets,
            criterion=self._config.criterion,
            class_count=self._class_count_value(),
        )

        leaf_node = DecisionTreeNode(
            node_id=node_id,
            depth=depth,
            prediction=prediction,
            sample_count=int(targets.shape[0]),
            class_counts=class_counts,
            impurity=impurity,
        )

        if self._should_stop(targets, depth=depth):
            return leaf_node

        split_evaluation = self._find_best_valid_split(features, targets)

        if split_evaluation is None:
            return leaf_node

        candidate = split_evaluation.candidate
        left_mask = features[:, candidate.feature_index] <= candidate.threshold
        right_mask = ~left_mask

        left = self._build_node(
            features[left_mask],
            targets[left_mask],
            depth=depth + 1,
        )
        right = self._build_node(
            features[right_mask],
            targets[right_mask],
            depth=depth + 1,
        )

        return DecisionTreeNode(
            node_id=node_id,
            depth=depth,
            prediction=prediction,
            sample_count=int(targets.shape[0]),
            class_counts=class_counts,
            impurity=impurity,
            split_evaluation=split_evaluation,
            left=left,
            right=right,
        )

    def _should_stop(self, targets: IntArray, *, depth: int) -> bool:
        """Return whether a node should become a leaf."""
        if depth >= self._config.max_depth:
            return True

        if targets.shape[0] < self._config.min_samples_split:
            return True

        return np.unique(targets).size == 1

    def _find_best_valid_split(
        self,
        features: FloatArray,
        targets: IntArray,
    ) -> SplitEvaluation | None:
        """Find the best split satisfying tree constraints."""
        candidates = generate_split_candidates(features)
        evaluations: list[SplitEvaluation] = []

        for candidate in candidates:
            try:
                evaluation = evaluate_split(
                    features,
                    targets,
                    candidate,
                    criterion=self._config.criterion,
                )
            except ValueError:
                continue

            if evaluation.left_sample_count < self._config.min_samples_leaf:
                continue

            if evaluation.right_sample_count < self._config.min_samples_leaf:
                continue

            if evaluation.information_gain < self._config.min_information_gain:
                continue

            evaluations.append(evaluation)

        if not evaluations:
            return None

        return max(
            evaluations,
            key=lambda evaluation: (
                evaluation.information_gain,
                -evaluation.candidate.feature_index,
                -evaluation.candidate.threshold,
            ),
        )

    def _predict_one(self, row: FloatArray, node: DecisionTreeNode) -> int:
        """Predict one sample by traversing the tree."""
        current = node

        while not current.is_leaf:
            assert current.split_evaluation is not None
            assert current.left is not None
            assert current.right is not None

            candidate = current.split_evaluation.candidate

            if row[candidate.feature_index] <= candidate.threshold:
                current = current.left
            else:
                current = current.right

        return current.prediction

    def _allocate_node_id(self) -> int:
        """Allocate a new node identifier."""
        node_id = self._next_node_id
        self._next_node_id += 1

        return node_id

    def _class_count_value(self) -> int:
        """Return the known number of classes."""
        assert self._class_count is not None

        return self._class_count

    def _ensure_fitted(self) -> None:
        """Ensure that the tree has been fitted."""
        if self._features is None or self._targets is None or self._root is None:
            msg = "The decision tree must be reset with a dataset before use."
            raise RuntimeError(msg)


def _extract_arrays(dataset: DatasetSnapshot) -> tuple[FloatArray, IntArray]:
    """Extract and validate features and labels from a dataset."""
    if dataset.targets is None:
        msg = "Dataset targets are required for decision tree training."
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
    if targets.ndim != EXPECTED_TARGET_DIMENSIONS:
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


def _compute_impurity(
    labels: IntArray,
    *,
    criterion: str,
    class_count: int,
) -> float:
    """Compute node impurity."""
    if criterion == "gini":
        return gini_impurity(labels, class_count=class_count)

    if criterion == "entropy":
        return entropy_impurity(labels, class_count=class_count)

    msg = f"Unsupported impurity criterion: {criterion}."
    raise ValueError(msg)


def _class_counts(labels: IntArray, *, class_count: int) -> tuple[int, ...]:
    """Return class counts as a tuple."""
    counts = np.bincount(labels, minlength=class_count)

    return tuple(int(value) for value in counts.tolist())


def _majority_class(labels: IntArray) -> int:
    """Return majority class label."""
    return int(np.argmax(np.bincount(labels)))


def _accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_true == y_pred))


def _iter_nodes(root: DecisionTreeNode) -> list[DecisionTreeNode]:
    """Return all tree nodes in pre-order."""
    nodes = [root]

    if root.left is not None:
        nodes.extend(_iter_nodes(root.left))

    if root.right is not None:
        nodes.extend(_iter_nodes(root.right))

    return nodes


def _count_nodes(root: DecisionTreeNode) -> int:
    """Count all nodes in the tree."""
    return len(_iter_nodes(root))


def _count_leaves(root: DecisionTreeNode) -> int:
    """Count leaf nodes in the tree."""
    if root.is_leaf:
        return 1

    assert root.left is not None
    assert root.right is not None

    return _count_leaves(root.left) + _count_leaves(root.right)


def _max_depth(root: DecisionTreeNode) -> int:
    """Return maximum depth of the tree."""
    if root.is_leaf:
        return root.depth

    assert root.left is not None
    assert root.right is not None

    return max(_max_depth(root.left), _max_depth(root.right))


def _validate_config(config: DecisionTreeConfig) -> None:
    """Validate decision tree configuration."""
    if config.max_depth < MIN_MAX_DEPTH:
        msg = "max_depth must be greater than or equal to 1."
        raise ValueError(msg)

    if config.min_samples_split < MIN_SAMPLES_SPLIT_LOWER_BOUND:
        msg = "min_samples_split must be greater than or equal to 2."
        raise ValueError(msg)

    if config.min_samples_leaf < MIN_SAMPLES_LEAF_LOWER_BOUND:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)

    if config.min_information_gain < 0.0:
        msg = "min_information_gain cannot be negative."
        raise ValueError(msg)
