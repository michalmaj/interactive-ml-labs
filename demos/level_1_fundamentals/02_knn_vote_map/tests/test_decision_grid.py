"""Tests for k-NN decision background grid."""

import numpy as np
import pytest
from knn_vote_map import compute_decision_grid

GRID_RESOLUTION: int = 8
K_ONE: int = 1


def _simple_features() -> np.ndarray:
    """Create a small deterministic two-class dataset."""
    return np.array(
        [
            [-2.0, 0.0],
            [-2.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )


def _simple_targets() -> np.ndarray:
    """Create labels for the simple dataset."""
    return np.array([0, 0, 1, 1], dtype=int)


def test_decision_grid_has_expected_shape() -> None:
    """Decision grid should have expected coordinate and label shapes."""
    grid = compute_decision_grid(
        _simple_features(),
        _simple_targets(),
        k=K_ONE,
        resolution=GRID_RESOLUTION,
    )

    assert grid.x_values.shape == (GRID_RESOLUTION,)
    assert grid.y_values.shape == (GRID_RESOLUTION,)
    assert grid.labels.shape == (GRID_RESOLUTION, GRID_RESOLUTION)


def test_decision_grid_contains_known_labels() -> None:
    """Decision grid labels should come from training labels."""
    grid = compute_decision_grid(
        _simple_features(),
        _simple_targets(),
        k=K_ONE,
        resolution=GRID_RESOLUTION,
    )

    assert set(np.unique(grid.labels).tolist()) == {0, 1}


def test_decision_grid_predicts_left_and_right_regions() -> None:
    """Left side should be class 0 and right side should be class 1."""
    grid = compute_decision_grid(
        _simple_features(),
        _simple_targets(),
        k=K_ONE,
        resolution=GRID_RESOLUTION,
    )

    middle_y_index = GRID_RESOLUTION // 2

    assert grid.labels[middle_y_index, 0] == 0
    assert grid.labels[middle_y_index, -1] == 1


@pytest.mark.parametrize(
    "features, targets, k, resolution, expected_message",
    [
        (
            np.array([1.0, 2.0]),
            np.array([0]),
            1,
            GRID_RESOLUTION,
            "two-dimensional",
        ),
        (
            np.empty((0, 2)),
            np.array([]),
            1,
            GRID_RESOLUTION,
            "cannot be empty",
        ),
        (
            np.array([[1.0, 2.0, 3.0]]),
            np.array([0]),
            1,
            GRID_RESOLUTION,
            "exactly two columns",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([[0]]),
            1,
            GRID_RESOLUTION,
            "one-dimensional",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0]),
            1,
            GRID_RESOLUTION,
            "same number of samples",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([0]),
            0,
            GRID_RESOLUTION,
            "k must be greater than 0",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([0]),
            2,
            GRID_RESOLUTION,
            "k cannot be greater",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([0]),
            1,
            1,
            "resolution must be at least 2",
        ),
    ],
)
def test_decision_grid_rejects_invalid_inputs(
    features: np.ndarray,
    targets: np.ndarray,
    k: int,
    resolution: int,
    expected_message: str,
) -> None:
    """Invalid decision grid inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        compute_decision_grid(
            features,
            targets,
            k=k,
            resolution=resolution,
        )
