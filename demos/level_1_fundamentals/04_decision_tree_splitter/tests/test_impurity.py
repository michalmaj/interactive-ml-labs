"""Tests for decision-tree impurity metrics."""

import numpy as np
import pytest
from decision_tree_splitter import (
    class_counts,
    class_probabilities,
    entropy_impurity,
    gini_impurity,
)

BALANCED_BINARY_GINI: float = 0.5
BALANCED_BINARY_ENTROPY: float = 1.0
IMBALANCED_BINARY_GINI: float = 0.375
IMBALANCED_BINARY_ENTROPY: float = 0.8112781244591328
PURE_NODE_IMPURITY: float = 0.0
CLASS_COUNT_WITH_MISSING_CLASS: int = 3


def test_class_counts_counts_binary_labels() -> None:
    """Class counts should count labels by class index."""
    labels = np.array([0, 1, 1, 0, 1])

    result = class_counts(labels)

    np.testing.assert_array_equal(result, np.array([2, 3]))


def test_class_counts_preserves_missing_classes_when_class_count_is_given() -> None:
    """Class counts should include zero counts for missing classes."""
    labels = np.array([1, 1])

    result = class_counts(labels, class_count=CLASS_COUNT_WITH_MISSING_CLASS)

    np.testing.assert_array_equal(result, np.array([0, 2, 0]))


def test_class_probabilities_computes_probabilities() -> None:
    """Class probabilities should sum to one."""
    labels = np.array([0, 1, 1, 0])

    result = class_probabilities(labels)

    np.testing.assert_allclose(result, np.array([0.5, 0.5]))
    assert np.sum(result) == pytest.approx(1.0)


def test_gini_impurity_is_zero_for_pure_node() -> None:
    """Pure node should have zero Gini impurity."""
    labels = np.array([1, 1, 1, 1])

    result = gini_impurity(labels)

    assert result == pytest.approx(PURE_NODE_IMPURITY)


def test_entropy_impurity_is_zero_for_pure_node() -> None:
    """Pure node should have zero entropy impurity."""
    labels = np.array([1, 1, 1, 1])

    result = entropy_impurity(labels)

    assert result == pytest.approx(PURE_NODE_IMPURITY)


def test_gini_impurity_for_balanced_binary_node() -> None:
    """Balanced binary node should have Gini impurity equal to 0.5."""
    labels = np.array([0, 0, 1, 1])

    result = gini_impurity(labels)

    assert result == pytest.approx(BALANCED_BINARY_GINI)


def test_entropy_impurity_for_balanced_binary_node() -> None:
    """Balanced binary node should have entropy equal to 1.0."""
    labels = np.array([0, 0, 1, 1])

    result = entropy_impurity(labels)

    assert result == pytest.approx(BALANCED_BINARY_ENTROPY)


def test_gini_impurity_for_imbalanced_binary_node() -> None:
    """Imbalanced binary node should have lower Gini than balanced node."""
    labels = np.array([0, 0, 0, 1])

    result = gini_impurity(labels)

    assert result == pytest.approx(IMBALANCED_BINARY_GINI)


def test_entropy_impurity_for_imbalanced_binary_node() -> None:
    """Imbalanced binary node should have lower entropy than balanced node."""
    labels = np.array([0, 0, 0, 1])

    result = entropy_impurity(labels)

    assert result == pytest.approx(IMBALANCED_BINARY_ENTROPY)


def test_impurity_metrics_support_multiclass_labels() -> None:
    """Impurity metrics should support more than two classes."""
    labels = np.array([0, 1, 2])

    gini = gini_impurity(labels)
    entropy = entropy_impurity(labels)

    assert gini == pytest.approx(2.0 / 3.0)
    assert entropy == pytest.approx(np.log2(3.0))


@pytest.mark.parametrize(
    "labels, expected_message",
    [
        (
            np.array([[0, 1]]),
            "one-dimensional",
        ),
        (
            np.array([]),
            "cannot be empty",
        ),
        (
            np.array([0.0, 1.0]),
            "integers",
        ),
        (
            np.array([0, -1]),
            "negative",
        ),
    ],
)
def test_class_counts_rejects_invalid_labels(
    labels: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid labels should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        class_counts(labels)


@pytest.mark.parametrize(
    "class_count, expected_message",
    [
        (
            0,
            "class_count must be greater than 0",
        ),
        (
            1,
            "class_count must be greater than the maximum label value",
        ),
    ],
)
def test_class_counts_rejects_invalid_class_count(
    class_count: int,
    expected_message: str,
) -> None:
    """Invalid class_count should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        class_counts(np.array([0, 1]), class_count=class_count)
