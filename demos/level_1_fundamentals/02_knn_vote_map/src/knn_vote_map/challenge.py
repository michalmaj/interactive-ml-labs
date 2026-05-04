"""Challenge mode for the k-NN Vote Map demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

type IntArray = NDArray[np.int_]

DEFAULT_TARGET_ACCURACY: Final[float] = 0.9

STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"


@dataclass(frozen=True, slots=True)
class KNNAccuracyChallengeConfig:
    """Configuration for an accuracy-based k-NN challenge.

    Attributes:
        target_accuracy: Accuracy value required to complete the challenge.
    """

    target_accuracy: float = DEFAULT_TARGET_ACCURACY


@dataclass(frozen=True, slots=True)
class KNNAccuracyChallengeResult:
    """Result of evaluating k-NN predictions against a hidden test set.

    Attributes:
        status: Challenge status.
        target_accuracy: Required accuracy.
        accuracy: Current test accuracy.
        correct_count: Number of correct predictions.
        sample_count: Number of evaluated test samples.
        message: Short explanation shown in the UI.
    """

    status: str
    target_accuracy: float
    accuracy: float
    correct_count: int
    sample_count: int
    message: str

    @property
    def success(self) -> bool:
        """Return whether the challenge has been completed successfully."""
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        """Return whether the challenge is currently failed."""
        return self.status == STATUS_FAILED


class KNNAccuracyChallenge:
    """Challenge requiring k-NN to reach target accuracy on a test set."""

    def __init__(self, config: KNNAccuracyChallengeConfig | None = None) -> None:
        """Initialize the challenge."""
        self._config = config or KNNAccuracyChallengeConfig()
        _validate_config(self._config)

    @property
    def config(self) -> KNNAccuracyChallengeConfig:
        """Return challenge configuration."""
        return self._config

    def evaluate(
        self,
        *,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> KNNAccuracyChallengeResult:
        """Evaluate predictions against the challenge target.

        Args:
            y_true: Expected labels.
            y_pred: Predicted labels.

        Returns:
            Challenge result containing accuracy and status.

        Raises:
            ValueError: If inputs are invalid.
        """
        true_labels = np.asarray(y_true, dtype=int)
        predicted_labels = np.asarray(y_pred, dtype=int)

        _validate_labels(true_labels, predicted_labels)

        correct_count = int(np.sum(true_labels == predicted_labels))
        sample_count = int(true_labels.shape[0])
        accuracy = correct_count / sample_count

        if accuracy >= self._config.target_accuracy:
            return KNNAccuracyChallengeResult(
                status=STATUS_SUCCESS,
                target_accuracy=self._config.target_accuracy,
                accuracy=accuracy,
                correct_count=correct_count,
                sample_count=sample_count,
                message="Challenge completed: test accuracy reached the target.",
            )

        return KNNAccuracyChallengeResult(
            status=STATUS_FAILED,
            target_accuracy=self._config.target_accuracy,
            accuracy=accuracy,
            correct_count=correct_count,
            sample_count=sample_count,
            message="Challenge not completed: try changing k, noise, or seed.",
        )


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Expected labels.
        y_pred: Predicted labels.

    Returns:
        Fraction of correct predictions.

    Raises:
        ValueError: If inputs are invalid.
    """
    true_labels = np.asarray(y_true, dtype=int)
    predicted_labels = np.asarray(y_pred, dtype=int)

    _validate_labels(true_labels, predicted_labels)

    return float(np.mean(true_labels == predicted_labels))


def _validate_config(config: KNNAccuracyChallengeConfig) -> None:
    """Validate challenge configuration."""
    if not 0.0 < config.target_accuracy <= 1.0:
        msg = "target_accuracy must be in the range (0, 1]."
        raise ValueError(msg)


def _validate_labels(y_true: IntArray, y_pred: IntArray) -> None:
    """Validate classification labels."""
    if y_true.ndim != 1 or y_pred.ndim != 1:
        msg = "y_true and y_pred must be one-dimensional arrays."
        raise ValueError(msg)

    if y_true.size == 0:
        msg = "y_true and y_pred cannot be empty."
        raise ValueError(msg)

    if y_true.shape != y_pred.shape:
        msg = f"y_true and y_pred must have the same shape. Got {y_true.shape} and {y_pred.shape}."
        raise ValueError(msg)
