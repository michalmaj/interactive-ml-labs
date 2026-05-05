"""Probability background grid for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from logistic_regression_boundary_lab.metrics import sigmoid

type FloatArray = NDArray[np.float64]

DEFAULT_GRID_RESOLUTION: Final[int] = 48
MIN_GRID_RESOLUTION: Final[int] = 2
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_FEATURE_COUNT: Final[int] = 2
EXPECTED_WEIGHT_DIMENSIONS: Final[int] = 1
MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.18


@dataclass(frozen=True, slots=True)
class ProbabilityGrid:
    """Positive-class probabilities over a regular 2D grid.

    Attributes:
        x_values: Grid x coordinates.
        y_values: Grid y coordinates.
        probabilities: Probability values with shape
            `(len(y_values), len(x_values))`.
    """

    x_values: FloatArray
    y_values: FloatArray
    probabilities: FloatArray


def compute_probability_grid(
    features: ArrayLike,
    *,
    weights: ArrayLike,
    bias: float,
    resolution: int = DEFAULT_GRID_RESOLUTION,
) -> ProbabilityGrid:
    """Compute positive-class probabilities over a regular 2D grid.

    Args:
        features: Two-dimensional reference feature matrix used to determine
            visualization bounds.
        weights: Logistic regression weight vector with two values.
        bias: Logistic regression bias.
        resolution: Number of grid points per axis.

    Returns:
        ProbabilityGrid containing x coordinates, y coordinates, and predicted
        probabilities for class `1`.

    Raises:
        ValueError: If inputs are invalid.
    """
    feature_array = np.asarray(features, dtype=float)
    weight_array = np.asarray(weights, dtype=float)

    _validate_inputs(
        features=feature_array,
        weights=weight_array,
        resolution=resolution,
    )

    x_min, x_max, y_min, y_max = _compute_bounds(feature_array)
    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)

    xx, yy = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    scores = grid_points @ weight_array + bias
    probabilities = sigmoid(scores).reshape(resolution, resolution)

    return ProbabilityGrid(
        x_values=x_values,
        y_values=y_values,
        probabilities=probabilities,
    )


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
    *,
    features: FloatArray,
    weights: FloatArray,
    resolution: int,
) -> None:
    """Validate probability grid inputs."""
    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        msg = "features must contain exactly two columns."
        raise ValueError(msg)

    if weights.ndim != EXPECTED_WEIGHT_DIMENSIONS:
        msg = "weights must be a one-dimensional array."
        raise ValueError(msg)

    if weights.shape[0] != EXPECTED_FEATURE_COUNT:
        msg = "weights must contain exactly two values."
        raise ValueError(msg)

    if resolution < MIN_GRID_RESOLUTION:
        msg = "resolution must be at least 2."
        raise ValueError(msg)
