"""Metrics used by the Gradient Descent Playground demo."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute the mean squared error between targets and predictions.

    Mean squared error is defined as the average squared difference between
    expected target values and predicted values.

    Args:
        y_true: Expected target values.
        y_pred: Predicted values.

    Returns:
        Mean squared error as a Python float.

    Raises:
        ValueError: If inputs have different shapes or contain no values.
    """
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)

    if true_values.shape != predicted_values.shape:
        msg = (
            "y_true and y_pred must have the same shape. "
            f"Got {true_values.shape} and {predicted_values.shape}."
        )
        raise ValueError(msg)

    if true_values.size == 0:
        msg = "y_true and y_pred cannot be empty."
        raise ValueError(msg)

    errors = true_values - predicted_values

    return float(np.mean(np.square(errors)))
