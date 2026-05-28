"""Challenge mode for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

DEFAULT_TARGET_TEST_ACCURACY: Final[float] = 0.85
DEFAULT_MAX_ROUND_COUNT: Final[int] = 8
DEFAULT_MAX_GENERALIZATION_GAP: Final[float] = 0.20

STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"


@dataclass(frozen=True, slots=True)
class BoostingChallengeConfig:
    """Configuration for Boosting Mistake Lab challenge mode.

    Attributes:
        target_test_accuracy: Minimum required boosted test accuracy.
        max_round_count: Maximum allowed number of boosting rounds.
        max_generalization_gap: Maximum allowed train-test accuracy gap.
    """

    target_test_accuracy: float = DEFAULT_TARGET_TEST_ACCURACY
    max_round_count: int = DEFAULT_MAX_ROUND_COUNT
    max_generalization_gap: float = DEFAULT_MAX_GENERALIZATION_GAP


@dataclass(frozen=True, slots=True)
class BoostingChallengeResult:
    """Result of challenge evaluation.

    Attributes:
        status: `success` or `failed`.
        passed: Boolean success flag.
        boosted_test_accuracy: Actual boosted test accuracy.
        round_count: Actual number of completed boosting rounds.
        generalization_gap: Actual boosted train-test accuracy gap.
        target_test_accuracy: Required minimum test accuracy.
        max_round_count: Maximum allowed number of rounds.
        max_generalization_gap: Maximum allowed generalization gap.
        failed_reasons: Human-readable failure reasons.
        summary: Compact summary for CLI/UI.
    """

    status: str
    passed: bool
    boosted_test_accuracy: float
    round_count: int
    generalization_gap: float
    target_test_accuracy: float
    max_round_count: int
    max_generalization_gap: float
    failed_reasons: tuple[str, ...]
    summary: str


def evaluate_boosting_challenge(
    trainer_snapshot: AlgorithmSnapshot,
    config: BoostingChallengeConfig | None = None,
) -> BoostingChallengeResult:
    """Evaluate challenge mode for a trained boosted ensemble.

    Args:
        trainer_snapshot: Snapshot produced by `BoostingTrainer`.
        config: Optional challenge configuration.

    Returns:
        BoostingChallengeResult with status and feedback.
    """
    config = config or BoostingChallengeConfig()
    _validate_config(config)

    boosted_test_accuracy = float(trainer_snapshot.metrics["boosted_test_accuracy"])
    round_count = int(trainer_snapshot.metrics["completed_round_count"])
    generalization_gap = float(trainer_snapshot.metrics["boosted_generalization_gap"])

    failed_reasons = _build_failed_reasons(
        boosted_test_accuracy=boosted_test_accuracy,
        round_count=round_count,
        generalization_gap=generalization_gap,
        config=config,
    )
    passed = not failed_reasons
    status = STATUS_SUCCESS if passed else STATUS_FAILED

    return BoostingChallengeResult(
        status=status,
        passed=passed,
        boosted_test_accuracy=boosted_test_accuracy,
        round_count=round_count,
        generalization_gap=generalization_gap,
        target_test_accuracy=config.target_test_accuracy,
        max_round_count=config.max_round_count,
        max_generalization_gap=config.max_generalization_gap,
        failed_reasons=failed_reasons,
        summary=_build_summary(
            passed=passed,
            boosted_test_accuracy=boosted_test_accuracy,
            round_count=round_count,
            generalization_gap=generalization_gap,
            config=config,
        ),
    )


def _build_failed_reasons(
    *,
    boosted_test_accuracy: float,
    round_count: int,
    generalization_gap: float,
    config: BoostingChallengeConfig,
) -> tuple[str, ...]:
    """Build failure reasons for challenge feedback."""
    reasons: list[str] = []

    if boosted_test_accuracy < config.target_test_accuracy:
        reasons.append(
            "boosted test accuracy is below target "
            f"({boosted_test_accuracy:.3f} < {config.target_test_accuracy:.3f})",
        )

    if round_count > config.max_round_count:
        reasons.append(
            f"too many boosting rounds ({round_count} > {config.max_round_count})",
        )

    if generalization_gap > config.max_generalization_gap:
        reasons.append(
            "generalization gap is too high "
            f"({generalization_gap:.3f} > {config.max_generalization_gap:.3f})",
        )

    return tuple(reasons)


def _build_summary(
    *,
    passed: bool,
    boosted_test_accuracy: float,
    round_count: int,
    generalization_gap: float,
    config: BoostingChallengeConfig,
) -> str:
    """Build compact challenge summary."""
    if passed:
        return (
            "Challenge success: boosted ensemble reached "
            f"test accuracy {boosted_test_accuracy:.3f}, used {round_count} rounds, "
            f"and kept gap {generalization_gap:.3f}."
        )

    return (
        "Challenge failed: target is "
        f"test accuracy >= {config.target_test_accuracy:.3f}, "
        f"rounds <= {config.max_round_count}, "
        f"gap <= {config.max_generalization_gap:.3f}."
    )


def _validate_config(config: BoostingChallengeConfig) -> None:
    """Validate challenge configuration."""
    if not 0.0 <= config.target_test_accuracy <= 1.0:
        msg = "target_test_accuracy must be in the range [0, 1]."
        raise ValueError(msg)

    if config.max_round_count < 1:
        msg = "max_round_count must be greater than or equal to 1."
        raise ValueError(msg)

    if config.max_generalization_gap < 0.0:
        msg = "max_generalization_gap cannot be negative."
        raise ValueError(msg)
