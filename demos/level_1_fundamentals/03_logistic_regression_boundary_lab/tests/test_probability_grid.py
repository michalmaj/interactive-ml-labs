"""Tests for logistic regression probability background grid."""

import numpy as np
import pytest
from logistic_regression_boundary_lab import compute_probability_grid

GRID_RESOLUTION: int = 8
EXPECTED_FEATURE_COUNT: int = 2


def _simple_features() -> np.ndarray:
    """Create a simple two-dimensional feature matrix."""
    return np.array(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [2.0, -1.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )


def test_probability_grid_has_expected_shape() -> None:
    """Probability grid should have expected coordinate and value shapes."""
    grid = compute_probability_grid(
        _simple_features(),
        weights=np.array([1.0, 0.0]),
        bias=0.0,
        resolution=GRID_RESOLUTION,
    )

    assert grid.x_values.shape == (GRID_RESOLUTION,)
    assert grid.y_values.shape == (GRID_RESOLUTION,)
    assert grid.probabilities.shape == (GRID_RESOLUTION, GRID_RESOLUTION)


def test_probability_grid_values_are_probabilities() -> None:
    """Grid values should be valid probabilities."""
    grid = compute_probability_grid(
        _simple_features(),
        weights=np.array([1.0, 0.0]),
        bias=0.0,
        resolution=GRID_RESOLUTION,
    )

    assert np.all(grid.probabilities >= 0.0)
    assert np.all(grid.probabilities <= 1.0)


def test_probability_grid_reflects_linear_score_direction() -> None:
    """With positive x weight, right side should have higher probability."""
    grid = compute_probability_grid(
        _simple_features(),
        weights=np.array([1.0, 0.0]),
        bias=0.0,
        resolution=GRID_RESOLUTION,
    )

    middle_y_index = GRID_RESOLUTION // 2
    left_probability = grid.probabilities[middle_y_index, 0]
    right_probability = grid.probabilities[middle_y_index, -1]

    assert left_probability < right_probability


def test_probability_grid_uses_bias() -> None:
    """Positive bias should increase probabilities for the same grid."""
    features = _simple_features()

    neutral_grid = compute_probability_grid(
        features,
        weights=np.array([0.0, 0.0]),
        bias=0.0,
        resolution=GRID_RESOLUTION,
    )
    biased_grid = compute_probability_grid(
        features,
        weights=np.array([0.0, 0.0]),
        bias=2.0,
        resolution=GRID_RESOLUTION,
    )

    assert np.mean(biased_grid.probabilities) > np.mean(neutral_grid.probabilities)


@pytest.mark.parametrize(
    "features, weights, resolution, expected_message",
    [
        (
            np.array([1.0, 2.0]),
            np.array([1.0, 0.0]),
            GRID_RESOLUTION,
            "two-dimensional",
        ),
        (
            np.empty((0, 2)),
            np.array([1.0, 0.0]),
            GRID_RESOLUTION,
            "features cannot be empty",
        ),
        (
            np.array([[1.0, 2.0, 3.0]]),
            np.array([1.0, 0.0]),
            GRID_RESOLUTION,
            "exactly two columns",
        ),
        (
            _simple_features(),
            np.array([[1.0, 0.0]]),
            GRID_RESOLUTION,
            "one-dimensional",
        ),
        (
            _simple_features(),
            np.array([1.0, 0.0, 0.0]),
            GRID_RESOLUTION,
            "exactly two values",
        ),
        (
            _simple_features(),
            np.array([1.0, 0.0]),
            1,
            "resolution must be at least 2",
        ),
    ],
)
def test_probability_grid_rejects_invalid_inputs(
    features: np.ndarray,
    weights: np.ndarray,
    resolution: int,
    expected_message: str,
) -> None:
    """Invalid probability grid inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        compute_probability_grid(
            features,
            weights=weights,
            bias=0.0,
            resolution=resolution,
        )
