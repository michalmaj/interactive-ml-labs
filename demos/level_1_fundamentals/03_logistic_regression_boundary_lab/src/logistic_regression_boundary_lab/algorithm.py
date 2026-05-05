"""Stepwise logistic regression for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot, MetricsHistory
from numpy.typing import ArrayLike, NDArray

from logistic_regression_boundary_lab.metrics import (
    binary_cross_entropy,
    classification_metrics,
    predict_labels_from_probabilities,
    sigmoid,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_LEARNING_RATE: Final[float] = 0.1
DEFAULT_INITIAL_WEIGHT_1: Final[float] = 0.0
DEFAULT_INITIAL_WEIGHT_2: Final[float] = 0.0
DEFAULT_INITIAL_BIAS: Final[float] = 0.0
DEFAULT_THRESHOLD: Final[float] = 0.5
DEFAULT_MAX_STEPS: Final[int] = 200
DEFAULT_CONVERGENCE_TOLERANCE: Final[float] = 1.0e-8

EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_FEATURE_COUNT: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfig:
    """Configuration for stepwise logistic regression.

    Attributes:
        learning_rate: Size of the parameter update step.
        initial_weight_1: Initial weight for feature `x1`.
        initial_weight_2: Initial weight for feature `x2`.
        initial_bias: Initial model bias.
        threshold: Decision threshold used to convert probabilities into labels.
        max_steps: Maximum number of gradient descent steps.
        convergence_tolerance: Small gradient norm threshold used to stop early.
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    initial_weight_1: float = DEFAULT_INITIAL_WEIGHT_1
    initial_weight_2: float = DEFAULT_INITIAL_WEIGHT_2
    initial_bias: float = DEFAULT_INITIAL_BIAS
    threshold: float = DEFAULT_THRESHOLD
    max_steps: int = DEFAULT_MAX_STEPS
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE


class StepwiseLogisticRegression:
    """Binary logistic regression trained step by step.

    The model computes a linear score:

    ```text
    score = w1 * x1 + w2 * x2 + bias
    ```

    Then it applies sigmoid to get the probability of class `1`.
    """

    name: str = "stepwise_logistic_regression"

    def __init__(self, config: LogisticRegressionConfig | None = None) -> None:
        """Initialize the model with optional configuration."""
        self._config = config or LogisticRegressionConfig()
        _validate_config(self._config)

        self._weights = np.array(
            [
                self._config.initial_weight_1,
                self._config.initial_weight_2,
            ],
            dtype=float,
        )
        self._bias = self._config.initial_bias
        self._iteration = 0
        self._done = False
        self._features: FloatArray | None = None
        self._targets: IntArray | None = None
        self._history = MetricsHistory()

    @property
    def weights(self) -> FloatArray:
        """Return a copy of the current model weights."""
        return self._weights.copy()

    @property
    def bias(self) -> float:
        """Return the current model bias."""
        return self._bias

    @property
    def threshold(self) -> float:
        """Return the current decision threshold."""
        return self._config.threshold

    @property
    def history(self) -> MetricsHistory:
        """Return metric history collected during training."""
        return self._history

    def reset(self, dataset: DatasetSnapshot) -> None:
        """Reset the model using a dataset snapshot.

        Args:
            dataset: Dataset containing two-dimensional features and binary targets.

        Raises:
            ValueError: If the dataset has invalid shape or labels.
        """
        features, targets = _extract_binary_classification_arrays(dataset)

        self._features = features
        self._targets = targets
        self._weights = np.array(
            [
                self._config.initial_weight_1,
                self._config.initial_weight_2,
            ],
            dtype=float,
        )
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

        probabilities = self._predict_probability_values(self._features)
        errors = probabilities - self._targets
        sample_count = float(self._features.shape[0])

        weight_gradients = (self._features.T @ errors) / sample_count
        bias_gradient = float(np.mean(errors))

        self._weights -= self._config.learning_rate * weight_gradients
        self._bias -= self._config.learning_rate * bias_gradient
        self._iteration += 1

        loss = self._compute_loss()
        gradient_norm = float(np.hypot(np.linalg.norm(weight_gradients), bias_gradient))

        self._history.add("loss", loss)
        self._history.add("weight_1_gradient", float(weight_gradients[0]))
        self._history.add("weight_2_gradient", float(weight_gradients[1]))
        self._history.add("bias_gradient", bias_gradient)
        self._history.add("gradient_norm", gradient_norm)

        self._done = (
            self._iteration >= self._config.max_steps
            or gradient_norm <= self._config.convergence_tolerance
        )

        return self._build_snapshot(
            status="finished" if self._done else "running",
            annotations=(
                f"Updated weights by {-self._config.learning_rate * weight_gradients}.",
                f"Updated bias by {-self._config.learning_rate * bias_gradient:.6f}.",
            ),
        )

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current model state without modifying it."""
        self._ensure_initialized()

        return self._build_snapshot(
            status="finished" if self._done else "running",
            annotations=("Current logistic regression state.",),
        )

    def predict_probabilities(self, features: ArrayLike) -> FloatArray:
        """Predict positive-class probabilities for input features."""
        values = _prepare_prediction_features(features)

        return self._predict_probability_values(values)

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict binary class labels for input features."""
        probabilities = self.predict_probabilities(features)

        return predict_labels_from_probabilities(
            probabilities,
            threshold=self._config.threshold,
        )

    def _predict_probability_values(self, features: FloatArray) -> FloatArray:
        """Predict positive-class probabilities without validating input shape."""
        scores = features @ self._weights + self._bias

        return sigmoid(scores)

    def _compute_loss(self) -> float:
        """Compute current binary cross-entropy loss."""
        assert self._features is not None
        assert self._targets is not None

        probabilities = self._predict_probability_values(self._features)

        return binary_cross_entropy(self._targets, probabilities)

    def _build_snapshot(
        self,
        *,
        status: str,
        annotations: tuple[str, ...],
    ) -> AlgorithmSnapshot:
        """Build an algorithm snapshot from the current model state."""
        assert self._features is not None
        assert self._targets is not None

        probabilities = self._predict_probability_values(self._features)
        predictions = predict_labels_from_probabilities(
            probabilities,
            threshold=self._config.threshold,
        )
        loss = binary_cross_entropy(self._targets, probabilities)
        metrics = classification_metrics(self._targets, predictions)

        return AlgorithmSnapshot(
            iteration=self._iteration,
            status=status,
            visual_state={
                "features": self._features,
                "targets": self._targets,
                "probabilities": probabilities,
                "predictions": predictions,
                "weights": self._weights.copy(),
                "bias": self._bias,
                "threshold": self._config.threshold,
                "loss_history": self._history.series("loss"),
            },
            metrics={
                "loss": loss,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "weight_1": float(self._weights[0]),
                "weight_2": float(self._weights[1]),
                "bias": self._bias,
                "threshold": self._config.threshold,
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


def _extract_binary_classification_arrays(
    dataset: DatasetSnapshot,
) -> tuple[FloatArray, IntArray]:
    """Extract and validate binary classification arrays from a dataset."""
    if dataset.targets is None:
        msg = "Dataset targets are required for binary classification."
        raise ValueError(msg)

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)

    _validate_training_features(features)
    _validate_training_targets(targets)

    if features.shape[0] != targets.shape[0]:
        msg = (
            "Dataset features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)

    return features, targets


def _prepare_prediction_features(features: ArrayLike) -> FloatArray:
    """Convert prediction input into a valid two-dimensional feature matrix."""
    values = np.asarray(features, dtype=float)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    _validate_training_features(values)

    return values


def _validate_training_features(features: FloatArray) -> None:
    """Validate feature matrix for the logistic regression demo."""
    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        msg = "features must contain exactly two columns."
        raise ValueError(msg)


def _validate_training_targets(targets: IntArray) -> None:
    """Validate binary target labels."""
    if targets.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = "targets must be a one-dimensional array."
        raise ValueError(msg)

    if targets.size == 0:
        msg = "targets cannot be empty."
        raise ValueError(msg)

    unique_labels = set(targets.tolist())

    if not unique_labels.issubset({0, 1}):
        msg = "targets must contain only binary labels 0 and 1."
        raise ValueError(msg)


def _validate_config(config: LogisticRegressionConfig) -> None:
    """Validate logistic regression configuration."""
    if config.learning_rate <= 0.0:
        msg = "learning_rate must be greater than 0."
        raise ValueError(msg)

    if not 0.0 <= config.threshold <= 1.0:
        msg = "threshold must be in the range [0, 1]."
        raise ValueError(msg)

    if config.max_steps <= 0:
        msg = "max_steps must be greater than 0."
        raise ValueError(msg)

    if config.convergence_tolerance < 0.0:
        msg = "convergence_tolerance cannot be negative."
        raise ValueError(msg)
