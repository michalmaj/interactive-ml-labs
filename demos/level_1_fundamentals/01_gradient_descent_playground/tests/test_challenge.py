"""Tests for Gradient Descent Playground challenge mode."""

import pytest
from gradient_descent_playground import LossChallenge, LossChallengeConfig
from ml_lab_core import AlgorithmSnapshot


def _snapshot(*, loss: float, iteration: int) -> AlgorithmSnapshot:
    """Create a minimal algorithm snapshot for challenge tests."""
    return AlgorithmSnapshot(
        iteration=iteration,
        metrics={"loss": loss},
    )


def test_loss_challenge_is_active_before_target_or_limit() -> None:
    """Challenge should stay active before success or failure."""
    challenge = LossChallenge(LossChallengeConfig(target_loss=1.0, max_steps=10))

    result = challenge.evaluate(_snapshot(loss=5.0, iteration=3))

    assert result.status == "active"
    assert result.success is False
    assert result.failed is False
    assert result.steps_remaining == 7


def test_loss_challenge_succeeds_when_target_loss_is_reached() -> None:
    """Challenge should succeed when current loss is low enough."""
    challenge = LossChallenge(LossChallengeConfig(target_loss=1.0, max_steps=10))

    result = challenge.evaluate(_snapshot(loss=0.8, iteration=4))

    assert result.status == "success"
    assert result.success is True
    assert result.failed is False
    assert result.steps_remaining == 6


def test_loss_challenge_fails_when_step_limit_is_reached() -> None:
    """Challenge should fail when max steps are reached without success."""
    challenge = LossChallenge(LossChallengeConfig(target_loss=1.0, max_steps=10))

    result = challenge.evaluate(_snapshot(loss=2.0, iteration=10))

    assert result.status == "failed"
    assert result.success is False
    assert result.failed is True
    assert result.steps_remaining == 0


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            LossChallengeConfig(target_loss=0.0),
            "target_loss must be greater than 0",
        ),
        (
            LossChallengeConfig(max_steps=0),
            "max_steps must be greater than 0",
        ),
    ],
)
def test_loss_challenge_rejects_invalid_config(
    config: LossChallengeConfig,
    expected_message: str,
) -> None:
    """Invalid challenge configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        LossChallenge(config)
