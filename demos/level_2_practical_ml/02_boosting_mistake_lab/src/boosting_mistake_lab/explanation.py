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
    language: str = "en",
) -> BoostingExplanation:
    """Build a short explanation for the current UI state.

    Args:
        trainer_snapshot: Snapshot produced by the boosting trainer.
        selected_stage: Currently selected boosting stage in the UI.
        confidence_view_enabled: Whether confidence view is enabled.
        challenge_result: Optional precomputed challenge result.
        language: UI language code. English is used as a fallback.

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

    normalized_language = _normalize_language(language)
    title = _title(challenge.passed, normalized_language)
    status = STATUS_SUCCESS if challenge.passed else STATUS_FAILED
    messages = _build_messages(
        selected_stage=selected_stage,
        round_count=round_count,
        best_round=best_round,
        best_test_accuracy=best_test_accuracy,
        confidence_view_enabled=confidence_view_enabled,
        language=normalized_language,
    )
    hints = _build_hints(
        challenge=challenge,
        selected_stage=selected_stage,
        best_round=best_round,
        language=normalized_language,
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
            language=normalized_language,
        ),
    )


def _build_messages(
    *,
    selected_stage: int,
    round_count: int,
    best_round: int,
    best_test_accuracy: float,
    confidence_view_enabled: bool,
    language: str,
) -> tuple[str, ...]:
    """Build explanation messages."""
    if language == "pl":
        stage_message = (
            f"Etap {selected_stage}/{round_count}: prawy panel korzysta z weak learnerów "
            f"z rund 1..{selected_stage}."
        )
        best_round_message = (
            f"Najlepsza staged test accuracy wynosi {best_test_accuracy:.3f} "
            f"w rundzie {best_round}."
        )
    else:
        stage_message = (
            f"Stage {selected_stage}/{round_count}: the right panel uses weak learners "
            f"from rounds 1..{selected_stage}."
        )
        best_round_message = (
            f"Best staged test accuracy is {best_test_accuracy:.3f} at round {best_round}."
        )
    confidence_message = _confidence_message(confidence_view_enabled, language)

    return stage_message, best_round_message, confidence_message


def _confidence_message(confidence_view_enabled: bool, language: str) -> str:
    """Build confidence-view explanation."""
    if language == "pl":
        if confidence_view_enabled:
            return (
                "Confidence view jest włączony: jaśniejsze regiony oznaczają "
                "słabszą zgodność w ensemble."
            )

        return "Confidence view jest wyłączony: kolory pokazują tylko predykcje klas."

    if confidence_view_enabled:
        return "Confidence view is on: pale regions mean weaker ensemble agreement."

    return "Confidence view is off: colors show only predicted classes."


def _build_hints(
    *,
    challenge: BoostingChallengeResult,
    selected_stage: int,
    best_round: int,
    language: str,
) -> tuple[str, ...]:
    """Build actionable hints."""
    hints: list[str] = []

    _append_hint(
        hints,
        challenge.boosted_test_accuracy < challenge.target_test_accuracy,
        language=language,
        en="Try increasing rounds, lowering noise, or inspecting the best staged round.",
        pl=(
            "Spróbuj zwiększyć liczbę rund, zmniejszyć szum albo obejrzeć "
            "najlepszą rundę na staged plot."
        ),
    )
    _append_hint(
        hints,
        challenge.round_count > challenge.max_round_count,
        language=language,
        en="Try using fewer total rounds and stop near the best staged test accuracy.",
        pl=("Spróbuj użyć mniejszej liczby rund i zatrzymać się blisko najlepszej test accuracy."),
    )
    _append_hint(
        hints,
        challenge.generalization_gap > challenge.max_generalization_gap,
        language=language,
        en="Try fewer rounds or a larger min_samples_leaf to reduce overfitting.",
        pl=(
            "Spróbuj mniejszej liczby rund albo większego min_samples_leaf, "
            "żeby ograniczyć overfitting."
        ),
    )

    if selected_stage < best_round:
        _append_hint(
            hints,
            True,
            language=language,
            en="Selected stage is before the best test round; move forward with Up.",
            pl=("Wybrany etap jest przed najlepszą rundą testową; przejdź dalej klawiszem Up."),
        )
    elif selected_stage > best_round:
        _append_hint(
            hints,
            True,
            language=language,
            en=(
                "Selected stage is after the best test round; compare whether test accuracy drops."
            ),
            pl=(
                "Wybrany etap jest po najlepszej rundzie testowej; "
                "sprawdź, czy test accuracy spada."
            ),
        )

    _append_hint(
        hints,
        not hints,
        language=language,
        en="Challenge passed. Try increasing noise or reducing rounds for a harder task.",
        pl=("Cel osiągnięty. Zwiększ szum albo zmniejsz liczbę rund, żeby utrudnić zadanie."),
    )

    return tuple(hints)


def _build_summary(
    *,
    challenge: BoostingChallengeResult,
    selected_stage: int,
    best_round: int,
    language: str,
) -> str:
    """Build compact one-line summary."""
    if challenge.passed:
        if language == "pl":
            return (
                f"Dobry balans: test={challenge.boosted_test_accuracy:.3f}, "
                f"gap={challenge.generalization_gap:.3f}, "
                f"rundy={challenge.round_count}."
            )

        return (
            f"Good balance: test={challenge.boosted_test_accuracy:.3f}, "
            f"gap={challenge.generalization_gap:.3f}, "
            f"rounds={challenge.round_count}."
        )

    if language == "pl":
        return (
            f"Jeszcze nie: wybrany etap {selected_stage}, najlepsza runda testowa {best_round}. "
            "Skorzystaj ze wskazówki niżej."
        )

    return (
        f"Not solved yet: selected stage {selected_stage}, best test round {best_round}. "
        "Use the hint below to tune the model."
    )


def _title(passed: bool, language: str) -> str:
    """Return localized explanation title."""
    if language == "pl":
        return "Cel osiągnięty" if passed else "Trzeba dostroić model"

    return "Challenge passed" if passed else "Challenge needs tuning"


def _normalize_language(language: str) -> str:
    """Return a supported UI language code."""
    if language.lower().startswith("pl"):
        return "pl"

    return "en"


def _append_hint(
    hints: list[str],
    condition: bool,
    *,
    language: str,
    en: str,
    pl: str,
) -> None:
    """Append one localized hint when its condition is true."""
    if not condition:
        return

    hints.append(pl if language == "pl" else en)


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
