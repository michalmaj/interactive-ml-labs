"""Human-readable explanations for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

from boosting_mistake_lab.challenge import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    BoostingChallengeResult,
    evaluate_boosting_challenge,
)

MIN_SELECTED_STAGE: Final[int] = 1


@dataclass(frozen=True, slots=True)
class BoostingExplanation:
    """Human-readable explanation for the current boosting state.

    Attributes:
        title: Short explanation title.
        status: Explanation status.
        messages: Main explanation messages.
        hints: Suggested actions.
        summary: Compact one-line summary for UI.
    """

    title: str
    status: str
    messages: tuple[str, ...]
    hints: tuple[str, ...]
    summary: str


def build_boosting_explanation(
    *,
    trainer_snapshot: AlgorithmSnapshot,
    selected_stage: int,
    confidence_view_enabled: bool,
    challenge_result: BoostingChallengeResult | None = None,
) -> BoostingExplanation:
    """Build a short explanation for the current UI state.

    Args:
        trainer_snapshot: Snapshot produced by the boosting trainer.
        selected_stage: Currently selected boosting stage in the UI.
        confidence_view_enabled: Whether confidence view is enabled.
        challenge_result: Optional precomputed challenge result.

    Returns:
        BoostingExplanation for CLI/UI rendering.

    Raises:
        ValueError: If selected stage is outside the trained round range.
    """
    challenge = challenge_result or evaluate_boosting_challenge(trainer_snapshot)
    round_count = int(trainer_snapshot.metrics["completed_round_count"])
    best_round = int(trainer_snapshot.metrics["best_staged_round_index"])
    best_test_accuracy = float(
        trainer_snapshot.metrics["best_staged_boosted_test_accuracy"],
    )

    _validate_selected_stage(selected_stage=selected_stage, round_count=round_count)

    title = "Challenge passed" if challenge.passed else "Challenge needs tuning"
    status = STATUS_SUCCESS if challenge.passed else STATUS_FAILED
    messages = _build_messages(
        selected_stage=selected_stage,
        round_count=round_count,
        best_round=best_round,
        best_test_accuracy=best_test_accuracy,
        confidence_view_enabled=confidence_view_enabled,
    )
    hints = _build_hints(
        challenge=challenge,
        selected_stage=selected_stage,
        best_round=best_round,
    )

    return BoostingExplanation(
        title=title,
        status=status,
        messages=messages,
        hints=hints,
        summary=_build_summary(
            challenge=challenge,
            selected_stage=selected_stage,
            best_round=best_round,
        ),
    )


def _build_messages(
    *,
    selected_stage: int,
    round_count: int,
    best_round: int,
    best_test_accuracy: float,
    confidence_view_enabled: bool,
) -> tuple[str, ...]:
    """Build explanation messages."""
    stage_message = (
        f"Stage {selected_stage}/{round_count}: the right panel uses weak learners "
        f"from rounds 1..{selected_stage}."
    )
    best_round_message = (
        f"Best staged test accuracy is {best_test_accuracy:.3f} at round {best_round}."
    )
    confidence_message = _confidence_message(confidence_view_enabled)

    return stage_message, best_round_message, confidence_message


def _confidence_message(confidence_view_enabled: bool) -> str:
    """Build confidence-view explanation."""
    if confidence_view_enabled:
        return "Confidence view is on: pale regions mean weaker ensemble agreement."

    return "Confidence view is off: colors show only predicted classes."


def _build_hints(
    *,
    challenge: BoostingChallengeResult,
    selected_stage: int,
    best_round: int,
) -> tuple[str, ...]:
    """Build actionable hints."""
    hints: list[str] = []

    if challenge.boosted_test_accuracy < challenge.target_test_accuracy:
        hints.append(
            "Try increasing rounds, lowering noise, or inspecting the best staged round.",
        )

    if challenge.round_count > challenge.max_round_count:
        hints.append(
            "Try using fewer total rounds and stop near the best staged test accuracy.",
        )

    if challenge.generalization_gap > challenge.max_generalization_gap:
        hints.append(
            "Try fewer rounds or a larger min_samples_leaf to reduce overfitting.",
        )

    if selected_stage < best_round:
        hints.append(
            "Selected stage is before the best test round; move forward with Up.",
        )
    elif selected_stage > best_round:
        hints.append(
            "Selected stage is after the best test round; compare whether test accuracy drops.",
        )

    if not hints:
        hints.append(
            "Challenge passed. Try increasing noise or reducing rounds for a harder task.",
        )

    return tuple(hints)


def _build_summary(
    *,
    challenge: BoostingChallengeResult,
    selected_stage: int,
    best_round: int,
) -> str:
    """Build compact one-line summary."""
    if challenge.passed:
        return (
            f"Good balance: test={challenge.boosted_test_accuracy:.3f}, "
            f"gap={challenge.generalization_gap:.3f}, "
            f"rounds={challenge.round_count}."
        )

    return (
        f"Not solved yet: selected stage {selected_stage}, best test round {best_round}. "
        "Use the hint below to tune the model."
    )


def _validate_selected_stage(*, selected_stage: int, round_count: int) -> None:
    """Validate selected stage range."""
    if selected_stage < MIN_SELECTED_STAGE:
        msg = "selected_stage must be greater than or equal to 1."
        raise ValueError(msg)

    if selected_stage > round_count:
        msg = (
            "selected_stage cannot be greater than completed round count. "
            f"Got {selected_stage} and {round_count}."
        )
        raise ValueError(msg)
