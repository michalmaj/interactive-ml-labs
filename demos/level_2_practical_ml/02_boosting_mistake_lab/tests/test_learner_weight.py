"""Tests for learner weight computation."""

from math import log

import pytest
from boosting_mistake_lab import (
    LearnerWeightResult,
    compute_learner_weight,
)

ERROR_ONE_QUARTER: float = 0.25
ERROR_ONE_HALF: float = 0.5
ERROR_THREE_QUARTERS: float = 0.75
EXPECTED_ALPHA_ONE_QUARTER: float = 0.5 * log(3.0)
EXPECTED_ALPHA_THREE_QUARTERS: float = -0.5 * log(3.0)
CUSTOM_EPSILON: float = 1.0e-6


def test_compute_learner_weight_returns_result() -> None:
    """Learner weight computation should return a result object."""
    result = compute_learner_weight(weighted_error=ERROR_ONE_QUARTER)

    assert isinstance(result, LearnerWeightResult)


def test_compute_learner_weight_is_positive_for_error_below_half() -> None:
    """A learner better than random guessing should get positive alpha."""
    result = compute_learner_weight(weighted_error=ERROR_ONE_QUARTER)

    assert result.learner_weight == pytest.approx(EXPECTED_ALPHA_ONE_QUARTER)
    assert result.learner_weight > 0.0
    assert result.weighted_error == pytest.approx(ERROR_ONE_QUARTER)
    assert result.clipped_weighted_error == pytest.approx(ERROR_ONE_QUARTER)


def test_compute_learner_weight_is_zero_for_error_equal_half() -> None:
    """A learner equivalent to random guessing should get alpha near zero."""
    result = compute_learner_weight(weighted_error=ERROR_ONE_HALF)

    assert result.learner_weight == pytest.approx(0.0)


def test_compute_learner_weight_is_negative_for_error_above_half() -> None:
    """A learner worse than random guessing should get negative alpha."""
    result = compute_learner_weight(weighted_error=ERROR_THREE_QUARTERS)

    assert result.learner_weight == pytest.approx(EXPECTED_ALPHA_THREE_QUARTERS)
    assert result.learner_weight < 0.0


def test_compute_learner_weight_clips_zero_error() -> None:
    """Zero error should be clipped to avoid infinite alpha."""
    result = compute_learner_weight(
        weighted_error=0.0,
        epsilon=CUSTOM_EPSILON,
    )

    expected_alpha = 0.5 * log((1.0 - CUSTOM_EPSILON) / CUSTOM_EPSILON)

    assert result.clipped_weighted_error == pytest.approx(CUSTOM_EPSILON)
    assert result.learner_weight == pytest.approx(expected_alpha)
    assert result.learner_weight > 0.0


def test_compute_learner_weight_clips_one_error() -> None:
    """Error equal to 1 should be clipped to avoid infinite negative alpha."""
    result = compute_learner_weight(
        weighted_error=1.0,
        epsilon=CUSTOM_EPSILON,
    )

    expected_alpha = 0.5 * log(CUSTOM_EPSILON / (1.0 - CUSTOM_EPSILON))

    assert result.clipped_weighted_error == pytest.approx(1.0 - CUSTOM_EPSILON)
    assert result.learner_weight == pytest.approx(expected_alpha)
    assert result.learner_weight < 0.0


@pytest.mark.parametrize(
    "weighted_error",
    [
        -0.1,
        1.1,
    ],
)
def test_compute_learner_weight_rejects_invalid_weighted_error(
    weighted_error: float,
) -> None:
    """Weighted error must be in the range [0, 1]."""
    with pytest.raises(ValueError, match="weighted_error"):
        compute_learner_weight(weighted_error=weighted_error)


@pytest.mark.parametrize(
    "epsilon",
    [
        0.0,
        0.5,
        0.75,
    ],
)
def test_compute_learner_weight_rejects_invalid_epsilon(epsilon: float) -> None:
    """Epsilon must be in the range (0, 0.5)."""
    with pytest.raises(ValueError, match="epsilon"):
        compute_learner_weight(
            weighted_error=ERROR_ONE_QUARTER,
            epsilon=epsilon,
        )
