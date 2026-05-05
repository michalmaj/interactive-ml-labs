"""Tests for Logistic Regression Boundary Lab challenge mode."""

import numpy as np
import pytest
from logistic_regression_boundary_lab import (
    PrecisionRecallChallenge,
    PrecisionRecallChallengeConfig,
)


def test_precision_recall_challenge_succeeds_when_targets_are_reached() -> None:
    """Challenge should succeed when both precision and recall reach targets."""
    challenge = PrecisionRecallChallenge(
        PrecisionRecallChallengeConfig(
            target_precision=0.6,
            target_recall=0.75,
        ),
    )

    result = challenge.evaluate(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
    )

    assert result.status == "success"
    assert result.success is True
    assert result.failed is False
    assert result.precision == pytest.approx(2.0 / 3.0)
    assert result.recall == pytest.approx(1.0)
    assert result.sample_count == 4


def test_precision_recall_challenge_fails_when_precision_is_too_low() -> None:
    """Challenge should fail when precision is below target."""
    challenge = PrecisionRecallChallenge(
        PrecisionRecallChallengeConfig(
            target_precision=0.9,
            target_recall=0.9,
        ),
    )

    result = challenge.evaluate(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 1]),
    )

    assert result.status == "failed"
    assert result.success is False
    assert result.failed is True
    assert result.precision == pytest.approx(2.0 / 3.0)
    assert result.recall == pytest.approx(1.0)


def test_precision_recall_challenge_fails_when_recall_is_too_low() -> None:
    """Challenge should fail when recall is below target."""
    challenge = PrecisionRecallChallenge(
        PrecisionRecallChallengeConfig(
            target_precision=0.9,
            target_recall=0.9,
        ),
    )

    result = challenge.evaluate(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 0]),
    )

    assert result.status == "failed"
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.5)


def test_precision_recall_challenge_exposes_confusion_matrix_counts() -> None:
    """Challenge result should expose TP, TN, FP, and FN."""
    challenge = PrecisionRecallChallenge(
        PrecisionRecallChallengeConfig(
            target_precision=0.5,
            target_recall=0.5,
        ),
    )

    result = challenge.evaluate(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 1, 1, 0]),
    )

    assert result.true_positive == 1
    assert result.true_negative == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.sample_count == 4


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            PrecisionRecallChallengeConfig(target_precision=0.0),
            "target_precision",
        ),
        (
            PrecisionRecallChallengeConfig(target_precision=1.1),
            "target_precision",
        ),
        (
            PrecisionRecallChallengeConfig(target_recall=0.0),
            "target_recall",
        ),
        (
            PrecisionRecallChallengeConfig(target_recall=1.1),
            "target_recall",
        ),
    ],
)
def test_precision_recall_challenge_rejects_invalid_config(
    config: PrecisionRecallChallengeConfig,
    expected_message: str,
) -> None:
    """Invalid challenge configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        PrecisionRecallChallenge(config)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_message",
    [
        (
            np.array([[0, 1]]),
            np.array([[0, 1]]),
            "one-dimensional",
        ),
        (
            np.array([]),
            np.array([]),
            "cannot be empty",
        ),
        (
            np.array([0, 2]),
            np.array([0, 1]),
            "binary labels",
        ),
        (
            np.array([0, 1]),
            np.array([0]),
            "same shape",
        ),
    ],
)
def test_precision_recall_challenge_rejects_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid challenge inputs should fail clearly."""
    challenge = PrecisionRecallChallenge()

    with pytest.raises(ValueError, match=expected_message):
        challenge.evaluate(y_true=y_true, y_pred=y_pred)
