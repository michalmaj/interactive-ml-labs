"""Decision background grid for the k-NN Vote Map demo."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from knn_vote_map.metrics import euclidean_distances

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_GRID_RESOLUTION: Final[int] = 48
MIN_GRID_RESOLUTION: Final[int] = 2
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_FEATURE_COUNT: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1
WORLD_MARGIN_RATIO: Final[float] = 0.18
MIN_WORLD_SPAN: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class DecisionGrid:
    """Predicted classes over a regular 2D grid.

    Attributes:
        x_values: Grid x coordinates.
        y_values: Grid y coordinates.
        labels: Predicted class labels with shape `(len(y_values), len(x_values))`.
    """

    x_values: FloatArray
    y_values: FloatArray
    labels: IntArray


def compute_decision_grid(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    k: int,
    resolution: int = DEFAULT_GRID_RESOLUTION,
) -> DecisionGrid:
    """Compute k-NN predictions over a regular 2D grid.

    The grid is used only for visualization. It shows which class would be
    predicted in different regions of the 2D space.

    Args:
        features: Two-dimensional training points.
        targets: One-dimensional class labels.
        k: Number of nearest neighbors used for voting.
        resolution: Number of grid points per axis.

    Returns:
        DecisionGrid containing x coordinates, y coordinates, and predicted labels.

    Raises:
        ValueError: If input data or configuration is invalid.
    """
    feature_array = np.asarray(features, dtype=float)
    target_array = np.asarray(targets, dtype=int)

    _validate_inputs(feature_array, target_array, k=k, resolution=resolution)

    x_min, x_max, y_min, y_max = _compute_bounds(feature_array)
    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)

    labels = np.empty((resolution, resolution), dtype=int)

    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            query_point = np.array([x_value, y_value], dtype=float)
            labels[y_index, x_index] = _predict_label(
                features=feature_array,
                targets=target_array,
                query_point=query_point,
                k=k,
            )

    return DecisionGrid(
        x_values=x_values,
        y_values=y_values,
        labels=labels,
    )


def _predict_label(
    *,
    features: FloatArray,
    targets: IntArray,
    query_point: FloatArray,
    k: int,
) -> int:
    """Predict one label without mutating the interactive classifier state."""
    distances = euclidean_distances(features, query_point)
    neighbor_indices = np.argsort(distances)[:k]
    neighbor_labels = [int(targets[index]) for index in neighbor_indices]

    return _select_winning_label(neighbor_labels)


def _select_winning_label(labels: list[int]) -> int:
    """Select the winning label using deterministic tie-breaking."""
    if not labels:
        msg = "Cannot select a class without labels."
        raise ValueError(msg)

    vote_counts = Counter(labels)
    max_votes = max(vote_counts.values())
    tied_labels = [label for label, count in vote_counts.items() if count == max_votes]

    return min(tied_labels)


def _compute_bounds(features: FloatArray) -> tuple[float, float, float, float]:
    """Compute visualization bounds with a small margin."""
    x_min = float(np.min(features[:, 0]))
    x_max = float(np.max(features[:, 0]))
    y_min = float(np.min(features[:, 1]))
    y_max = float(np.max(features[:, 1]))

    x_span = max(x_max - x_min, MIN_WORLD_SPAN)
    y_span = max(y_max - y_min, MIN_WORLD_SPAN)

    x_margin = x_span * WORLD_MARGIN_RATIO
    y_margin = y_span * WORLD_MARGIN_RATIO

    return (
        x_min - x_margin,
        x_max + x_margin,
        y_min - y_margin,
        y_max + y_margin,
    )


def _validate_inputs(
    features: FloatArray,
    targets: IntArray,
    *,
    k: int,
    resolution: int,
) -> None:
    """Validate inputs for decision grid computation."""
    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        msg = "features must contain exactly two columns."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    if targets.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = "targets must be a one-dimensional array."
        raise ValueError(msg)

    if features.shape[0] != targets.shape[0]:
        msg = (
            "features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)

    if k <= 0:
        msg = "k must be greater than 0."
        raise ValueError(msg)

    if k > features.shape[0]:
        msg = (
            "k cannot be greater than the number of samples. "
            f"Got k={k} and sample_count={features.shape[0]}."
        )
        raise ValueError(msg)

    if resolution < MIN_GRID_RESOLUTION:
        msg = "resolution must be at least 2."
        raise ValueError(msg)
