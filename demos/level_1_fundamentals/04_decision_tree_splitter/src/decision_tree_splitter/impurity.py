"""Impurity metrics for the Decision Tree Splitter demo."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

EXPECTED_LABEL_DIMENSIONS: Final[int] = 1
MIN_CLASS_COUNT: Final[int] = 1


def class_counts(labels: ArrayLike, *, class_count: int | None = None) -> IntArray:
    """Count samples assigned to each class.

    Args:
        labels: One-dimensional integer class labels.
        class_count: Optional total number of classes. When provided, missing
            classes are represented with zero counts.

    Returns:
        Integer array containing class counts.

    Raises:
        ValueError: If labels or class_count are invalid.
    """
    label_values = _validate_labels(labels)
    resolved_class_count = _resolve_class_count(
        label_values,
        class_count=class_count,
    )

    return np.bincount(label_values, minlength=resolved_class_count).astype(int)


def class_probabilities(labels: ArrayLike, *, class_count: int | None = None) -> FloatArray:
    """Compute class probabilities from labels.

    Args:
        labels: One-dimensional integer class labels.
        class_count: Optional total number of classes.

    Returns:
        Probability of each class in the node.

    Raises:
        ValueError: If labels or class_count are invalid.
    """
    counts = class_counts(labels, class_count=class_count)
    sample_count = float(np.sum(counts))

    return counts.astype(float) / sample_count


def gini_impurity(labels: ArrayLike, *, class_count: int | None = None) -> float:
    """Compute Gini impurity for a node.

    Gini impurity is equal to zero when all samples in the node belong to the
    same class. Higher values mean that classes are more mixed.

    Args:
        labels: One-dimensional integer class labels.
        class_count: Optional total number of classes.

    Returns:
        Gini impurity value.

    Raises:
        ValueError: If labels or class_count are invalid.
    """
    probabilities = class_probabilities(labels, class_count=class_count)

    return float(1.0 - np.sum(probabilities**2))


def entropy_impurity(labels: ArrayLike, *, class_count: int | None = None) -> float:
    """Compute entropy impurity for a node.

    Entropy is equal to zero when all samples in the node belong to the same
    class. For a balanced binary node, entropy is equal to one.

    Args:
        labels: One-dimensional integer class labels.
        class_count: Optional total number of classes.

    Returns:
        Entropy impurity value using base-2 logarithm.

    Raises:
        ValueError: If labels or class_count are invalid.
    """
    probabilities = class_probabilities(labels, class_count=class_count)
    positive_probabilities = probabilities[probabilities > 0.0]

    return float(-np.sum(positive_probabilities * np.log2(positive_probabilities)))


def _validate_labels(labels: ArrayLike) -> IntArray:
    """Validate and convert labels into an integer NumPy array."""
    label_values = np.asarray(labels)

    if label_values.ndim != EXPECTED_LABEL_DIMENSIONS:
        msg = "labels must be a one-dimensional array."
        raise ValueError(msg)

    if label_values.size == 0:
        msg = "labels cannot be empty."
        raise ValueError(msg)

    if not np.issubdtype(label_values.dtype, np.integer):
        msg = "labels must contain integers."
        raise ValueError(msg)

    integer_labels = label_values.astype(int)

    if np.any(integer_labels < 0):
        msg = "labels cannot contain negative values."
        raise ValueError(msg)

    return integer_labels


def _resolve_class_count(labels: IntArray, *, class_count: int | None) -> int:
    """Resolve and validate class count."""
    if class_count is None:
        return int(np.max(labels)) + 1

    if class_count < MIN_CLASS_COUNT:
        msg = "class_count must be greater than 0."
        raise ValueError(msg)

    max_label = int(np.max(labels))

    if max_label >= class_count:
        msg = (
            "class_count must be greater than the maximum label value. "
            f"Got class_count={class_count} and max_label={max_label}."
        )
        raise ValueError(msg)

    return class_count
