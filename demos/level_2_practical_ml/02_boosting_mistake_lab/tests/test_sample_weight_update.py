"""Tests for sample weight update utilities."""

from math import exp, inf, nan

import numpy as np
import pytest
from boosting_mistake_lab import (
    SampleWeightUpdateResult,
    update_sample_weights,
)

LEARNER_WEIGHT: float = 0.5
UNIFORM_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)


def test_update_sample_weights_returns_result() -> None:
    """Sample weight update should return a result object."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=LEARNER_WEIGHT,
    )

    assert isinstance(result, SampleWeightUpdateResult)


def test_update_sample_weights_normalizes_updated_weights() -> None:
    """Updated sample weights should sum to 1.0."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=LEARNER_WEIGHT,
    )

    assert np.sum(result.updated_weights) == pytest.approx(1.0)


def test_update_sample_weights_increases_mistake_weights_for_positive_alpha() -> None:
    """Positive learner weight should increase relative importance of mistakes."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=LEARNER_WEIGHT,
    )

    assert result.updated_weights[1] > result.old_weights[1]
    assert result.updated_weights[3] > result.old_weights[3]
    assert result.updated_weights[0] < result.old_weights[0]
    assert result.updated_weights[2] < result.old_weights[2]


def test_update_sample_weights_computes_expected_values() -> None:
    """Updated weights should follow exp(+alpha) / exp(-alpha) multipliers."""
    mistake_multiplier = exp(LEARNER_WEIGHT)
    correct_multiplier = exp(-LEARNER_WEIGHT)

    unnormalized = np.array(
        [
            0.25 * correct_multiplier,
            0.25 * mistake_multiplier,
            0.25 * correct_multiplier,
            0.25 * mistake_multiplier,
        ],
        dtype=float,
    )
    expected = unnormalized / np.sum(unnormalized)

    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=LEARNER_WEIGHT,
    )

    np.testing.assert_allclose(result.updated_weights, expected)
    assert result.mistake_multiplier == pytest.approx(mistake_multiplier)
    assert result.correct_multiplier == pytest.approx(correct_multiplier)


def test_update_sample_weights_preserves_weights_for_zero_alpha() -> None:
    """Zero learner weight should leave weights unchanged after normalization."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=0.0,
    )

    np.testing.assert_allclose(result.updated_weights, UNIFORM_WEIGHTS)


def test_update_sample_weights_reduces_mistake_weights_for_negative_alpha() -> None:
    """Negative learner weight should reduce relative importance of mistakes."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=-LEARNER_WEIGHT,
    )

    assert result.updated_weights[1] < result.old_weights[1]
    assert result.updated_weights[3] < result.old_weights[3]
    assert result.updated_weights[0] > result.old_weights[0]
    assert result.updated_weights[2] > result.old_weights[2]


def test_update_sample_weights_reports_weight_sums() -> None:
    """Update result should expose old and updated mistake/correct weight sums."""
    result = update_sample_weights(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 0, 1, 1]),
        sample_weights=UNIFORM_WEIGHTS,
        learner_weight=LEARNER_WEIGHT,
    )

    assert result.old_mistake_weight_sum == pytest.approx(0.5)
    assert result.old_correct_weight_sum == pytest.approx(0.5)
    assert result.updated_mistake_weight_sum == pytest.approx(
        np.sum(result.updated_weights[result.mistake_mask]),
    )
    assert result.updated_correct_weight_sum == pytest.approx(
        np.sum(result.updated_weights[result.correct_mask]),
    )


@pytest.mark.parametrize(
    "learner_weight",
    [
        nan,
        inf,
        -inf,
    ],
)
def test_update_sample_weights_rejects_non_finite_learner_weight(
    learner_weight: float,
) -> None:
    """Learner weight must be finite."""
    with pytest.raises(ValueError, match="learner_weight"):
        update_sample_weights(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            sample_weights=np.array([0.5, 0.5]),
            learner_weight=learner_weight,
        )


@pytest.mark.parametrize(
    "sample_weights, expected_message",
    [
        (
            np.array([0.5, -0.5]),
            "sample_weights cannot contain negative values",
        ),
        (
            np.array([0.2, 0.2]),
            "sample_weights must sum to 1.0",
        ),
        (
            np.array([1.0]),
            "sample_weights must contain the same number of samples",
        ),
    ],
)
def test_update_sample_weights_rejects_invalid_weights(
    sample_weights: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid sample weights should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        update_sample_weights(
            y_true=np.array([0, 1]),
            y_pred=np.array([0, 1]),
            sample_weights=sample_weights,
            learner_weight=LEARNER_WEIGHT,
        )
