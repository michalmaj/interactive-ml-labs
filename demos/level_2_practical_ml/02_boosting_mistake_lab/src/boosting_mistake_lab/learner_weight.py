"""Learner weight utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Final

DEFAULT_ERROR_EPSILON: Final[float] = 1.0e-12
MIN_WEIGHTED_ERROR: Final[float] = 0.0
MAX_WEIGHTED_ERROR: Final[float] = 1.0
MIN_EPSILON: Final[float] = 0.0
MAX_EPSILON: Final[float] = 0.5


@dataclass(frozen=True, slots=True)
class LearnerWeightResult:
    """Result of learner weight computation.

    Attributes:
        learner_weight: Learner contribution weight, commonly called alpha.
        weighted_error: Original weighted error before clipping.
        clipped_weighted_error: Weighted error after numerical clipping.
        epsilon: Numerical epsilon used for clipping.
    """

    learner_weight: float
    weighted_error: float
    clipped_weighted_error: float
    epsilon: float


def compute_learner_weight(
    *,
    weighted_error: float,
    epsilon: float = DEFAULT_ERROR_EPSILON,
) -> LearnerWeightResult:
    """Compute AdaBoost-style learner weight from weighted error.

    The classic AdaBoost learner weight is:

    ```text
    alpha = 0.5 * log((1 - error) / error)
    ```

    Directly using `error = 0` or `error = 1` would lead to infinite values.
    Therefore, the error is clipped to `[epsilon, 1 - epsilon]`.

    Args:
        weighted_error: Weighted classification error in the range [0, 1].
        epsilon: Numerical clipping value.

    Returns:
        LearnerWeightResult with original error, clipped error, and alpha.

    Raises:
        ValueError: If weighted_error or epsilon is invalid.
    """
    _validate_weighted_error(weighted_error)
    _validate_epsilon(epsilon)

    clipped_error = _clip_weighted_error(
        weighted_error=weighted_error,
        epsilon=epsilon,
    )
    learner_weight = 0.5 * log((1.0 - clipped_error) / clipped_error)

    return LearnerWeightResult(
        learner_weight=learner_weight,
        weighted_error=weighted_error,
        clipped_weighted_error=clipped_error,
        epsilon=epsilon,
    )


def _clip_weighted_error(*, weighted_error: float, epsilon: float) -> float:
    """Clip weighted error to avoid infinite learner weights."""
    return min(1.0 - epsilon, max(epsilon, weighted_error))


def _validate_weighted_error(weighted_error: float) -> None:
    """Validate weighted error."""
    if not MIN_WEIGHTED_ERROR <= weighted_error <= MAX_WEIGHTED_ERROR:
        msg = "weighted_error must be in the range [0, 1]."
        raise ValueError(msg)


def _validate_epsilon(epsilon: float) -> None:
    """Validate clipping epsilon."""
    if not MIN_EPSILON < epsilon < MAX_EPSILON:
        msg = "epsilon must be in the range (0, 0.5)."
        raise ValueError(msg)
