"""Challenge mode for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from logistic_regression_boundary_lab.metrics import (
    classification_metrics,
    confusion_matrix_counts,
)

type IntArray = NDArray[np.int_]

DEFAULT_TARGET_PRECISION: Final[float] = 0.80
DEFAULT_TARGET_RECALL: Final[float] = 0.90

STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"


@dataclass(frozen=True, slots=True)
class PrecisionRecallChallengeConfig:
    """Configuration for a precision-recall challenge.

    Attributes:
        target_precision: Minimum required precision.
        target_recall: Minimum required recall.
    """

    target_precision: float = DEFAULT_TARGET_PRECISION
    target_recall: float = DEFAULT_TARGET_RECALL


@dataclass(frozen=True, slots=True)
class PrecisionRecallChallengeResult:
    """Result of evaluating predictions against precision-recall targets.

    Attributes:
        status: Challenge status.
        target_precision: Required minimum precision.
        target_recall: Required minimum recall.
        accuracy: Current accuracy.
        precision: Current precision.
        recall: Current recall.
        true_positive: True positive count.
        true_negative: True negative count.
        false_positive: False positive count.
        false_negative: False negative count.
        sample_count: Number of evaluated samples.
        message: Short student-facing explanation.
    """

    status: str
    target_precision: float
    target_recall: float
    accuracy: float
    precision: float
    recall: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    sample_count: int
    message: str

    @property
    def success(self) -> bool:
        """Return whether the challenge target is satisfied."""
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        """Return whether the challenge target is not satisfied."""
        return self.status == STATUS_FAILED


class PrecisionRecallChallenge:
    """Challenge requiring both precision and recall to reach target values."""

    def __init__(self, config: PrecisionRecallChallengeConfig | None = None) -> None:
        """Initialize the challenge."""
        self._config = config or PrecisionRecallChallengeConfig()
        _validate_config(self._config)

    @property
    def config(self) -> PrecisionRecallChallengeConfig:
        """Return challenge configuration."""
        return self._config

    def evaluate(
        self,
        *,
        y_true: ArrayLike,
        y_pred: ArrayLike,
    ) -> PrecisionRecallChallengeResult:
        """Evaluate predicted labels against precision and recall targets.

        Args:
            y_true: Expected binary labels.
            y_pred: Predicted binary labels.

        Returns:
            Challenge result containing classification metrics and status.

        Raises:
            ValueError: If inputs are invalid.
        """
        true_labels = np.asarray(y_true, dtype=int)
        predicted_labels = np.asarray(y_pred, dtype=int)

        metrics = classification_metrics(true_labels, predicted_labels)
        counts = confusion_matrix_counts(true_labels, predicted_labels)

        precision_ok = metrics.precision >= self._config.target_precision
        recall_ok = metrics.recall >= self._config.target_recall

        if precision_ok and recall_ok:
            return PrecisionRecallChallengeResult(
                status=STATUS_SUCCESS,
                target_precision=self._config.target_precision,
                target_recall=self._config.target_recall,
                accuracy=metrics.accuracy,
                precision=metrics.precision,
                recall=metrics.recall,
                true_positive=counts.true_positive,
                true_negative=counts.true_negative,
                false_positive=counts.false_positive,
                false_negative=counts.false_negative,
                sample_count=int(true_labels.shape[0]),
                message="Challenge completed: precision and recall reached their targets.",
            )

        return PrecisionRecallChallengeResult(
            status=STATUS_FAILED,
            target_precision=self._config.target_precision,
            target_recall=self._config.target_recall,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            true_positive=counts.true_positive,
            true_negative=counts.true_negative,
            false_positive=counts.false_positive,
            false_negative=counts.false_negative,
            sample_count=int(true_labels.shape[0]),
            message="Challenge not completed: tune threshold, learning rate, or noise.",
        )


def _validate_config(config: PrecisionRecallChallengeConfig) -> None:
    """Validate precision-recall challenge configuration."""
    if not 0.0 < config.target_precision <= 1.0:
        msg = "target_precision must be in the range (0, 1]."
        raise ValueError(msg)

    if not 0.0 < config.target_recall <= 1.0:
        msg = "target_recall must be in the range (0, 1]."
        raise ValueError(msg)
