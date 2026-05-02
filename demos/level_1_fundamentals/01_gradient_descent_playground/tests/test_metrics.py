"""Tests for Gradient Descent Playground metrics."""

import numpy as np
import pytest
from gradient_descent_playground import mean_squared_error

EXPECTED_ZERO_MSE: float = 0.0
EXPECTED_UNIT_MSE: float = 1.0
EXPECTED_MIXED_MSE: float = 5.0 / 3.0


def test_mean_squared_error_is_zero_for_identical_values() -> None:
    """MSE should be zero when predictions are exactly correct."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    result = mean_squared_error(y_true, y_pred)

    assert result == EXPECTED_ZERO_MSE


def test_mean_squared_error_computes_average_squared_error() -> None:
    """MSE should average squared differences."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])

    result = mean_squared_error(y_true, y_pred)

    assert result == EXPECTED_UNIT_MSE


def test_mean_squared_error_accepts_python_lists() -> None:
    """MSE should accept regular Python sequences."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 4.0, 4.0]

    result = mean_squared_error(y_true, y_pred)

    assert result == pytest.approx(EXPECTED_MIXED_MSE)


def test_mean_squared_error_rejects_different_shapes() -> None:
    """Inputs with different shapes should fail clearly."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([[1.0], [2.0], [3.0]])

    with pytest.raises(ValueError, match="same shape"):
        mean_squared_error(y_true, y_pred)


def test_mean_squared_error_rejects_empty_inputs() -> None:
    """Empty inputs should fail clearly."""
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError, match="cannot be empty"):
        mean_squared_error(y_true, y_pred)
