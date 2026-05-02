"""Stepwise linear regression trained with gradient descent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot, MetricsHistory
from numpy.typing import NDArray

from gradient_descent_playground.metrics import mean_squared_error

type FloatArray = NDArray[np.float64]

DEFAULT_LEARNING_RATE: Final[float] = 0.03
DEFAULT_INITIAL_WEIGHT: Final[float] = 0.0
DEFAULT_INITIAL_BIAS: Final[float] = 0.0
DEFAULT_MAX_STEPS: Final[int] = 100
DEFAULT_CONVERGENCE_TOLERANCE: Final[float] = 1.0e-10

EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_FEATURE_COLUMNS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class GradientDescentConfig:
    """Configuration for stepwise linear regression.

    Attributes:
        learning_rate: Size of the parameter update step.
        initial_weight: Initial slope of the regression line.
        initial_bias: Initial intercept of the regression line.
        max_steps: Maximum number of optimization steps.
        convergence_tolerance: Small gradient norm threshold used to stop early.
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    initial_weight: float = DEFAULT_INITIAL_WEIGHT
    initial_bias: float = DEFAULT_INITIAL_BIAS
    max_steps: int = DEFAULT_MAX_STEPS
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE


class StepwiseLinearRegression:
    """One-dimensional linear regression trained step by step.

    The model predicts values using:

    ```text
    y_pred = weight * x + bias
    ```

    Each call to :meth:`step` performs one gradient descent update for both
    parameters.
    """

    name: str = "linear_regression_gradient_descent"

    def __init__(self, config: GradientDescentConfig | None = None) -> None:
        """Initialize the model with optional gradient descent configuration."""
        self._config = config or GradientDescentConfig()
        _validate_config(self._config)

        self._weight = self._config.initial_weight
        self._bias = self._config.initial_bias
        self._iteration = 0
        self._done = False
        self._features: FloatArray | None = None
        self._targets: FloatArray | None = None
        self._history = MetricsHistory()

    @property
    def weight(self) -> float:
        """Return the current regression weight."""
        return self._weight

    @property
    def bias(self) -> float:
        """Return the current regression bias."""
        return self._bias

    @property
    def history(self) -> MetricsHistory:
        """Return metric history collected during training."""
        return self._history

    def reset(self, dataset: DatasetSnapshot) -> None:
        """Reset the model using a dataset snapshot.

        Args:
            dataset: Dataset containing one-dimensional features and targets.

        Raises:
            ValueError: If the dataset has invalid shape.
        """
        features, targets = _extract_regression_arrays(dataset)

        self._features = features
        self._targets = targets
        self._weight = self._config.initial_weight
        self._bias = self._config.initial_bias
        self._iteration = 0
        self._done = False
        self._history.reset()
        self._history.add("loss", self._compute_loss())

    def step(self) -> AlgorithmSnapshot:
        """Perform one gradient descent update and return the new snapshot."""
        self._ensure_initialized()

        if self._done:
            return self.snapshot()

        assert self._features is not None
        assert self._targets is not None

        predictions = self._predict_values(self._features)
        errors = predictions - self._targets

        sample_count = float(self._features.shape[0])
        x_values = self._features[:, 0]

        weight_gradient = float((2.0 / sample_count) * np.sum(errors * x_values))
        bias_gradient = float((2.0 / sample_count) * np.sum(errors))

        self._weight -= self._config.learning_rate * weight_gradient
        self._bias -= self._config.learning_rate * bias_gradient
        self._iteration += 1

        loss = self._compute_loss()
        gradient_norm = float(np.hypot(weight_gradient, bias_gradient))

        self._history.add("loss", loss)
        self._history.add("weight_gradient", weight_gradient)
        self._history.add("bias_gradient", bias_gradient)
        self._history.add("gradient_norm", gradient_norm)

        self._done = (
            self._iteration >= self._config.max_steps
            or gradient_norm <= self._config.convergence_tolerance
        )

        return self._build_snapshot(
            status="finished" if self._done else "running",
            annotations=(
                f"Updated weight by {-self._config.learning_rate * weight_gradient:.6f}.",
                f"Updated bias by {-self._config.learning_rate * bias_gradient:.6f}.",
            ),
        )

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current model state without modifying it."""
        self._ensure_initialized()

        return self._build_snapshot(
            status="finished" if self._done else "running",
            annotations=("Current model state.",),
        )

    def predict(self, features: FloatArray) -> FloatArray:
        """Predict target values for one-dimensional input features."""
        values = np.asarray(features, dtype=float)

        if values.ndim == 1:
            values = values.reshape(-1, EXPECTED_FEATURE_COLUMNS)

        if (
            values.ndim != EXPECTED_FEATURE_DIMENSIONS
            or values.shape[1] != EXPECTED_FEATURE_COLUMNS
        ):
            msg = (
                "features must be a one-dimensional array or a two-dimensional "
                "array with one column."
            )
            raise ValueError(msg)

        return self._predict_values(values)

    def _predict_values(self, features: FloatArray) -> FloatArray:
        """Predict target values without validating input shape."""
        return self._weight * features[:, 0] + self._bias

    def _compute_loss(self) -> float:
        """Compute current mean squared error."""
        assert self._features is not None
        assert self._targets is not None

        predictions = self._predict_values(self._features)

        return mean_squared_error(self._targets, predictions)

    def _build_snapshot(
        self,
        *,
        status: str,
        annotations: tuple[str, ...],
    ) -> AlgorithmSnapshot:
        """Build an algorithm snapshot from the current model state."""
        assert self._features is not None
        assert self._targets is not None

        predictions = self._predict_values(self._features)
        loss = mean_squared_error(self._targets, predictions)

        return AlgorithmSnapshot(
            iteration=self._iteration,
            status=status,
            visual_state={
                "features": self._features,
                "targets": self._targets,
                "predictions": predictions,
                "weight": self._weight,
                "bias": self._bias,
                "loss_history": self._history.series("loss"),
            },
            metrics={
                "loss": loss,
                "weight": self._weight,
                "bias": self._bias,
                "learning_rate": self._config.learning_rate,
            },
            annotations=annotations,
            done=self._done,
        )

    def _ensure_initialized(self) -> None:
        """Ensure that the model was initialized with a dataset."""
        if self._features is None or self._targets is None:
            msg = "The model must be reset with a dataset before use."
            raise RuntimeError(msg)


def _extract_regression_arrays(dataset: DatasetSnapshot) -> tuple[FloatArray, FloatArray]:
    """Extract and validate one-dimensional regression arrays from a dataset."""
    if dataset.targets is None:
        msg = "Dataset targets are required for regression."
        raise ValueError(msg)

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=float)

    if (
        features.ndim != EXPECTED_FEATURE_DIMENSIONS
        or features.shape[1] != EXPECTED_FEATURE_COLUMNS
    ):
        msg = "Dataset features must be a two-dimensional array with exactly one column."
        raise ValueError(msg)

    if targets.ndim != 1:
        msg = "Dataset targets must be a one-dimensional array."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "Dataset cannot be empty."
        raise ValueError(msg)

    if features.shape[0] != targets.shape[0]:
        msg = (
            "Dataset features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)

    return features, targets


def _validate_config(config: GradientDescentConfig) -> None:
    """Validate gradient descent configuration."""
    if config.learning_rate <= 0.0:
        msg = "learning_rate must be greater than 0."
        raise ValueError(msg)

    if config.max_steps <= 0:
        msg = "max_steps must be greater than 0."
        raise ValueError(msg)

    if config.convergence_tolerance < 0.0:
        msg = "convergence_tolerance cannot be negative."
        raise ValueError(msg)
