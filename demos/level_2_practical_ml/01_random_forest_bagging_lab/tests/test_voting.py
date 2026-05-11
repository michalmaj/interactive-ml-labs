"""Tests for majority voting utilities."""

import numpy as np
import pytest
from random_forest_bagging_lab import VotingResult, majority_vote

CLASS_COUNT: int = 2


def test_majority_vote_returns_voting_result() -> None:
    """Majority vote should return a VotingResult object."""
    tree_predictions = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=CLASS_COUNT)

    assert isinstance(result, VotingResult)


def test_majority_vote_computes_final_predictions() -> None:
    """Majority vote should select the most frequent class per sample."""
    tree_predictions = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=CLASS_COUNT)

    np.testing.assert_array_equal(result.predictions, np.array([0, 1, 1, 0]))


def test_majority_vote_computes_vote_counts() -> None:
    """Vote counts should be returned per sample and class."""
    tree_predictions = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=CLASS_COUNT)

    expected_vote_counts = np.array(
        [
            [2, 1],
            [0, 3],
            [1, 2],
            [3, 0],
        ],
        dtype=int,
    )

    np.testing.assert_array_equal(result.vote_counts, expected_vote_counts)


def test_majority_vote_computes_confidence() -> None:
    """Vote confidence should be the winning vote fraction."""
    tree_predictions = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=CLASS_COUNT)

    np.testing.assert_allclose(
        result.confidence,
        np.array([2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0]),
    )


def test_majority_vote_uses_lowest_class_label_as_tie_breaker() -> None:
    """Ties should be resolved by NumPy argmax, selecting the lowest class label."""
    tree_predictions = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=CLASS_COUNT)

    np.testing.assert_array_equal(result.predictions, np.array([0, 0]))
    np.testing.assert_allclose(result.confidence, np.array([0.5, 0.5]))


def test_majority_vote_infers_class_count_when_not_provided() -> None:
    """Class count should be inferred from predictions when omitted."""
    tree_predictions = np.array(
        [
            [0, 2],
            [2, 2],
            [2, 1],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions)

    assert result.vote_counts.shape == (2, 3)
    np.testing.assert_array_equal(result.predictions, np.array([2, 2]))


def test_majority_vote_preserves_missing_classes_when_class_count_is_given() -> None:
    """Vote counts should include zero columns for missing classes."""
    tree_predictions = np.array(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=int,
    )

    result = majority_vote(tree_predictions, class_count=3)

    expected_vote_counts = np.array(
        [
            [0, 2, 0],
            [0, 2, 0],
        ],
        dtype=int,
    )

    np.testing.assert_array_equal(result.vote_counts, expected_vote_counts)


@pytest.mark.parametrize(
    "tree_predictions, expected_message",
    [
        (
            np.array([0, 1, 1]),
            "two-dimensional",
        ),
        (
            np.empty((0, 3), dtype=int),
            "at least one tree",
        ),
        (
            np.empty((3, 0), dtype=int),
            "at least one sample",
        ),
        (
            np.array([[0.0, 1.0]]),
            "integers",
        ),
        (
            np.array([[0, -1]]),
            "negative",
        ),
    ],
)
def test_majority_vote_rejects_invalid_predictions(
    tree_predictions: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid prediction matrices should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        majority_vote(tree_predictions)


@pytest.mark.parametrize(
    "class_count, expected_message",
    [
        (
            0,
            "class_count must be greater than 0",
        ),
        (
            1,
            "class_count must be greater than the maximum predicted label",
        ),
    ],
)
def test_majority_vote_rejects_invalid_class_count(
    class_count: int,
    expected_message: str,
) -> None:
    """Invalid class_count should fail clearly."""
    tree_predictions = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=int,
    )

    with pytest.raises(ValueError, match=expected_message):
        majority_vote(tree_predictions, class_count=class_count)
