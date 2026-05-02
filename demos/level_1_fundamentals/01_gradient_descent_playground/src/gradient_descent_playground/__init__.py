"""Gradient Descent Playground demo package."""

from gradient_descent_playground.dataset import (
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)
from gradient_descent_playground.metrics import mean_squared_error

__all__ = [
    "SyntheticRegressionConfig",
    "make_synthetic_regression_dataset",
    "mean_squared_error",
]
