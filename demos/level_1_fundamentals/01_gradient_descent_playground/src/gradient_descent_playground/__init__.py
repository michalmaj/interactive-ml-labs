"""Gradient Descent Playground demo package."""

from gradient_descent_playground.algorithm import (
    GradientDescentConfig,
    StepwiseLinearRegression,
)
from gradient_descent_playground.challenge import (
    ChallengeResult,
    LossChallenge,
    LossChallengeConfig,
)
from gradient_descent_playground.dataset import (
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)
from gradient_descent_playground.metrics import mean_squared_error
from gradient_descent_playground.scene import GradientDescentScene

__all__ = [
    "ChallengeResult",
    "GradientDescentConfig",
    "GradientDescentScene",
    "LossChallenge",
    "LossChallengeConfig",
    "StepwiseLinearRegression",
    "SyntheticRegressionConfig",
    "make_synthetic_regression_dataset",
    "mean_squared_error",
]
