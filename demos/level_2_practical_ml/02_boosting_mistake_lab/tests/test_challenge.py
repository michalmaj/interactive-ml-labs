"""Tests for Boosting Mistake Lab challenge mode."""

import pytest
from boosting_mistake_lab import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    BoostingChallengeConfig,
    BoostingChallengeResult,
    evaluate_boosting_challenge,
)
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    boosted_test_accuracy: float,
    completed_round_count: int,
    boosted_generalization_gap: float,
) -> AlgorithmSnapshot:
    """Create a minimal trainer snapshot for challenge tests."""
    return AlgorithmSnapshot(
        iteration=completed_round_count,
        status="completed",
        metrics={
            "boosted_test_accuracy": boosted_test_accuracy,
            "completed_round_count": completed_round_count,
            "boosted_generalization_gap": boosted_generalization_gap,
        },
        visual_state={},
        annotations=(),
        done=True,
    )


def test_boosting_challenge_returns_result() -> None:
    """Challenge evaluation should return a result object."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
        ),
    )

    assert isinstance(result, BoostingChallengeResult)


def test_boosting_challenge_succeeds_when_all_targets_are_met() -> None:
    """Challenge should succeed when accuracy, rounds, and gap satisfy targets."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
        ),
    )

    assert result.status == STATUS_SUCCESS
    assert result.passed is True
    assert result.failed_reasons == ()


def test_boosting_challenge_fails_when_test_accuracy_is_too_low() -> None:
    """Challenge should fail when boosted test accuracy is below target."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.80,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
        ),
    )

    assert result.status == STATUS_FAILED
    assert result.passed is False
    assert any("test accuracy" in reason for reason in result.failed_reasons)


def test_boosting_challenge_fails_when_round_count_is_too_high() -> None:
    """Challenge should fail when too many boosting rounds are used."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=12,
            boosted_generalization_gap=0.10,
        ),
    )

    assert result.status == STATUS_FAILED
    assert result.passed is False
    assert any("too many" in reason for reason in result.failed_reasons)


def test_boosting_challenge_fails_when_gap_is_too_high() -> None:
    """Challenge should fail when generalization gap is above limit."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.30,
        ),
    )

    assert result.status == STATUS_FAILED
    assert result.passed is False
    assert any("gap" in reason for reason in result.failed_reasons)


def test_boosting_challenge_collects_multiple_failure_reasons() -> None:
    """Challenge should report all failed criteria."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.70,
            completed_round_count=12,
            boosted_generalization_gap=0.35,
        ),
    )

    assert result.status == STATUS_FAILED
    assert len(result.failed_reasons) == 3


def test_boosting_challenge_accepts_custom_config() -> None:
    """Challenge should accept custom targets."""
    result = evaluate_boosting_challenge(
        _snapshot(
            boosted_test_accuracy=0.76,
            completed_round_count=10,
            boosted_generalization_gap=0.25,
        ),
        BoostingChallengeConfig(
            target_test_accuracy=0.75,
            max_round_count=10,
            max_generalization_gap=0.25,
        ),
    )

    assert result.status == STATUS_SUCCESS
    assert result.passed is True


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            BoostingChallengeConfig(target_test_accuracy=-0.1),
            "target_test_accuracy",
        ),
        (
            BoostingChallengeConfig(target_test_accuracy=1.1),
            "target_test_accuracy",
        ),
        (
            BoostingChallengeConfig(max_round_count=0),
            "max_round_count",
        ),
        (
            BoostingChallengeConfig(max_generalization_gap=-0.1),
            "max_generalization_gap",
        ),
    ],
)
def test_boosting_challenge_rejects_invalid_config(
    config: BoostingChallengeConfig,
    expected_message: str,
) -> None:
    """Invalid challenge config should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        evaluate_boosting_challenge(
            _snapshot(
                boosted_test_accuracy=0.90,
                completed_round_count=5,
                boosted_generalization_gap=0.10,
            ),
            config,
        )
