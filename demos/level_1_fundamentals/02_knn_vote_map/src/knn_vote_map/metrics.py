"""Distance metrics used by the k-NN Vote Map demo."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]


def euclidean_distance(point_a: ArrayLike, point_b: ArrayLike) -> float:
    """Compute Euclidean distance between two points.

    Euclidean distance is the straight-line distance between two points.

    Args:
        point_a: First point.
        point_b: Second point.

    Returns:
        Euclidean distance as a Python float.

    Raises:
        ValueError: If points have different shapes or are empty.
    """
    first = np.asarray(point_a, dtype=float)
    second = np.asarray(point_b, dtype=float)

    _validate_points(first, second)

    difference = first - second

    return float(np.linalg.norm(difference))


def euclidean_distances(points: ArrayLike, query_point: ArrayLike) -> FloatArray:
    """Compute Euclidean distances from many points to one query point.

    Args:
        points: Two-dimensional array of reference points.
        query_point: One-dimensional query point.

    Returns:
        One-dimensional array of distances.

    Raises:
        ValueError: If input shapes are invalid.
    """
    reference_points = np.asarray(points, dtype=float)
    query = np.asarray(query_point, dtype=float)

    _validate_points_matrix(reference_points)
    _validate_query_point(query, expected_feature_count=reference_points.shape[1])

    differences = reference_points - query

    return np.linalg.norm(differences, axis=1)


def _validate_points(point_a: FloatArray, point_b: FloatArray) -> None:
    """Validate two individual points."""
    if point_a.size == 0 or point_b.size == 0:
        msg = "Points cannot be empty."
        raise ValueError(msg)

    if point_a.shape != point_b.shape:
        msg = f"Points must have the same shape. Got {point_a.shape} and {point_b.shape}."
        raise ValueError(msg)


def _validate_points_matrix(points: FloatArray) -> None:
    """Validate a matrix of reference points."""
    if points.ndim != 2:
        msg = "points must be a two-dimensional array."
        raise ValueError(msg)

    if points.shape[0] == 0:
        msg = "points cannot be empty."
        raise ValueError(msg)

    if points.shape[1] == 0:
        msg = "points must contain at least one feature."
        raise ValueError(msg)


def _validate_query_point(query_point: FloatArray, *, expected_feature_count: int) -> None:
    """Validate one query point against the expected feature count."""
    if query_point.ndim != 1:
        msg = "query_point must be a one-dimensional array."
        raise ValueError(msg)

    if query_point.size == 0:
        msg = "query_point cannot be empty."
        raise ValueError(msg)

    if query_point.shape[0] != expected_feature_count:
        msg = (
            "query_point must have the same number of features as points. "
            f"Got {query_point.shape[0]} and {expected_feature_count}."
        )
        raise ValueError(msg)
