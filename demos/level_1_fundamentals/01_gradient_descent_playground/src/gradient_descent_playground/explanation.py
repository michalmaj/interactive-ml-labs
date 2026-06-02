"""Explanation helpers for the Gradient Descent Playground demo."""

from __future__ import annotations

from typing import Final

from ml_lab_core import AlgorithmSnapshot

from gradient_descent_playground.challenge import ChallengeResult

MAX_EXPLANATION_LINES: Final[int] = 2
INITIAL_STEP: Final[int] = 0


def build_explanation_lines(
    snapshot: AlgorithmSnapshot,
    challenge_result: ChallengeResult,
    *,
    language: str = "en",
) -> tuple[str, ...]:
    """Build short student-facing explanation lines.

    The explanation panel should not repeat all internal details. It should give
    students a compact interpretation of the current algorithm state.

    Args:
        snapshot: Current algorithm snapshot.
        challenge_result: Current challenge evaluation result.

    Returns:
        Short explanation lines ready to be displayed by the renderer.
    """
    language = _normalize_language(language)

    if challenge_result.success:
        return _lines(
            language=language,
            en=(
                "Challenge completed: the model reached the target loss.",
                "Try increasing noise or lowering learning rate to make it harder.",
            ),
            pl=(
                "Cel osiągnięty: model zszedł poniżej target loss.",
                "Zwiększ szum albo obniż learning rate, żeby utrudnić zadanie.",
            ),
        )

    if challenge_result.failed:
        return _lines(
            language=language,
            en=(
                "Challenge failed: the step limit was reached before target loss.",
                "Try increasing learning rate carefully or reducing data noise.",
            ),
            pl=(
                "Cel nieosiągnięty: limit kroków skończył się przed target loss.",
                "Ostrożnie zwiększ learning rate albo zmniejsz szum danych.",
            ),
        )

    if snapshot.iteration == INITIAL_STEP:
        return _lines(
            language=language,
            en=(
                "The model starts with initial weight and bias.",
                "Press N for one step or Space to run gradient descent.",
            ),
            pl=(
                "Model startuje z początkową wagą i biasem.",
                "N robi jeden krok, a Space uruchamia gradient descent.",
            ),
        )

    annotation_lines = _annotation_lines(snapshot)

    if annotation_lines:
        return annotation_lines

    return _lines(
        language=language,
        en=(
            "The model is updating parameters to reduce mean squared error.",
            _current_state_line(snapshot, language=language),
        ),
        pl=(
            "Model aktualizuje parametry, żeby zmniejszać mean squared error.",
            _current_state_line(snapshot, language=language),
        ),
    )


def _annotation_lines(snapshot: AlgorithmSnapshot) -> tuple[str, ...]:
    """Return explanation lines based on algorithm annotations."""
    if not snapshot.annotations:
        return ()

    lines = tuple(str(annotation) for annotation in snapshot.annotations if annotation)

    if not lines:
        return ()

    if len(lines) == 1:
        return (
            lines[0],
            _current_state_line(snapshot),
        )

    return lines[:MAX_EXPLANATION_LINES]


def _current_state_line(snapshot: AlgorithmSnapshot, *, language: str = "en") -> str:
    """Build a compact line with current loss and parameters."""
    loss = float(snapshot.metrics["loss"])
    weight = float(snapshot.metrics["weight"])
    bias = float(snapshot.metrics["bias"])

    if _normalize_language(language) == "pl":
        return f"Teraz: loss={loss:.4f}, weight={weight:.3f}, bias={bias:.3f}."

    return f"Current loss: {loss:.4f}, weight: {weight:.3f}, bias: {bias:.3f}."


def _normalize_language(language: str) -> str:
    """Return a supported UI language code."""
    if language.lower().startswith("pl"):
        return "pl"

    return "en"


def _lines(
    *,
    language: str,
    en: tuple[str, str],
    pl: tuple[str, str],
) -> tuple[str, str]:
    """Return two localized explanation lines."""
    if language == "pl":
        return pl

    return en
