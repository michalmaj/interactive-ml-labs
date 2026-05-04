"""Tests for k-NN Vote Map distance metrics."""

import numpy as np
import pytest
from knn_vote_map import euclidean_distance, euclidean_distances

EXPECTED_ZERO_DISTANCE: float = 0.0
EXPECTED_THREE_FOUR_FIVE_DISTANCE: float = 5.0


def test_euclidean_distance_is_zero_for_identical_points() -> None:
    """Distance between identical points should be zero."""
    result = euclidean_distance([1.0, 2.0], [1.0, 2.0])

    assert result == EXPECTED_ZERO_DISTANCE


def test_euclidean_distance_uses_straight_line_distance() -> None:
    """Euclidean distance should follow the Pythagorean theorem."""
    result = euclidean_distance([0.0, 0.0], [3.0, 4.0])

    assert result == EXPECTED_THREE_FOUR_FIVE_DISTANCE


def test_euclidean_distance_accepts_numpy_arrays() -> None:
    """Euclidean distance should accept NumPy arrays."""
    point_a = np.array([1.0, 1.0])
    point_b = np.array([4.0, 5.0])

    result = euclidean_distance(point_a, point_b)

    assert result == EXPECTED_THREE_FOUR_FIVE_DISTANCE


def test_euclidean_distance_rejects_different_shapes() -> None:
    """Points with different shapes should fail clearly."""
    with pytest.raises(ValueError, match="same shape"):
        euclidean_distance([1.0, 2.0], [1.0, 2.0, 3.0])


def test_euclidean_distance_rejects_empty_points() -> None:
    """Empty points should fail clearly."""
    with pytest.raises(ValueError, match="cannot be empty"):
        euclidean_distance([], [])


def test_euclidean_distances_computes_distances_to_query_point() -> None:
    """Distances from many points to one query point should be computed."""
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
    )
    query_point = np.array([0.0, 0.0])

    result = euclidean_distances(points, query_point)

    np.testing.assert_allclose(result, np.array([0.0, 5.0, 10.0]))


def test_euclidean_distances_accepts_python_lists() -> None:
    """Distances function should accept regular Python sequences."""
    points = [
        [1.0, 1.0],
        [4.0, 5.0],
    ]
    query_point = [1.0, 1.0]

    result = euclidean_distances(points, query_point)

    np.testing.assert_allclose(result, np.array([0.0, 5.0]))


def test_euclidean_distances_rejects_one_dimensional_points() -> None:
    """Reference points must be represented as a matrix."""
    with pytest.raises(ValueError, match="two-dimensional"):
        euclidean_distances([1.0, 2.0, 3.0], [1.0])


def test_euclidean_distances_rejects_empty_points() -> None:
    """Reference points cannot be empty."""
    with pytest.raises(ValueError, match="points cannot be empty"):
        euclidean_distances(np.empty((0, 2)), [0.0, 0.0])


def test_euclidean_distances_rejects_points_without_features() -> None:
    """Reference points must contain at least one feature."""
    with pytest.raises(ValueError, match="at least one feature"):
        euclidean_distances(np.empty((3, 0)), [])


def test_euclidean_distances_rejects_non_vector_query_point() -> None:
    """Query point must be one-dimensional."""
    with pytest.raises(ValueError, match="one-dimensional"):
        euclidean_distances(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))


def test_euclidean_distances_rejects_empty_query_point() -> None:
    """Query point cannot be empty."""
    with pytest.raises(ValueError, match="query_point cannot be empty"):
        euclidean_distances(np.array([[1.0, 2.0]]), [])


def test_euclidean_distances_rejects_query_point_with_wrong_feature_count() -> None:
    """Query point must match the feature count of reference points."""
    with pytest.raises(ValueError, match="same number of features"):
        euclidean_distances(np.array([[1.0, 2.0]]), [1.0, 2.0, 3.0])
