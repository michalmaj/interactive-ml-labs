"""Tests for k-NN Vote Map challenge mode."""

import numpy as np
import pytest
from knn_vote_map import (
    KNNAccuracyChallenge,
    KNNAccuracyChallengeConfig,
    accuracy_score,
)


def test_accuracy_score_computes_fraction_of_correct_predictions() -> None:
    """Accuracy should be the fraction of matching labels."""
    result = accuracy_score(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
    )

    assert result == pytest.approx(0.75)


def test_accuracy_challenge_succeeds_when_target_is_reached() -> None:
    """Challenge should succeed when accuracy reaches the target."""
    challenge = KNNAccuracyChallenge(KNNAccuracyChallengeConfig(target_accuracy=0.75))

    result = challenge.evaluate(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
    )

    assert result.status == "success"
    assert result.success is True
    assert result.failed is False
    assert result.correct_count == 3
    assert result.sample_count == 4


def test_accuracy_challenge_fails_when_target_is_not_reached() -> None:
    """Challenge should fail when accuracy is below the target."""
    challenge = KNNAccuracyChallenge(KNNAccuracyChallengeConfig(target_accuracy=1.0))

    result = challenge.evaluate(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
    )

    assert result.status == "failed"
    assert result.success is False
    assert result.failed is True
    assert result.correct_count == 3
    assert result.sample_count == 4


@pytest.mark.parametrize(
    "target_accuracy",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_accuracy_challenge_rejects_invalid_target_accuracy(target_accuracy: float) -> None:
    """Target accuracy should be in the range (0, 1]."""
    with pytest.raises(ValueError, match="target_accuracy"):
        KNNAccuracyChallenge(KNNAccuracyChallengeConfig(target_accuracy=target_accuracy))


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
            np.array([0, 1]),
            np.array([0]),
            "same shape",
        ),
    ],
)
def test_accuracy_score_rejects_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid accuracy inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        accuracy_score(y_true, y_pred)
