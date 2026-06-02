"""Explanation helpers for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from typing import Final

from ml_lab_core import AlgorithmSnapshot

from logistic_regression_boundary_lab.challenge import PrecisionRecallChallengeResult

MAX_EXPLANATION_LINES: Final[int] = 2


def build_explanation_lines(
    snapshot: AlgorithmSnapshot,
    challenge_result: PrecisionRecallChallengeResult,
    *,
    language: str = "en",
) -> tuple[str, ...]:
    """Build short student-facing explanation lines.

    Args:
        snapshot: Current logistic regression snapshot.
        challenge_result: Current precision-recall challenge result.

    Returns:
        Short explanation lines ready to be displayed by the renderer.
    """
    language = _normalize_language(language)

    if snapshot.iteration == 0:
        return _initial_lines(challenge_result, language=language)

    model_line = _model_line(snapshot, language=language)
    challenge_line = _challenge_line(snapshot, challenge_result, language=language)

    return (model_line, challenge_line)[:MAX_EXPLANATION_LINES]


def _initial_lines(
    challenge_result: PrecisionRecallChallengeResult,
    *,
    language: str,
) -> tuple[str, ...]:
    """Build explanation lines before training starts."""
    if challenge_result.success:
        if language == "pl":
            return (
                "Cel jest już osiągnięty na ukrytych danych testowych.",
                "Trenuj krok po kroku i obserwuj, jak zmienia się tło probability.",
            )

        return (
            "Challenge is already completed on hidden test data.",
            "Train step by step and observe how probability background changes.",
        )

    if language == "pl":
        return (
            "Tło pokazuje probability klasy class_1 przed thresholdingiem.",
            "Trenuj model, a potem dostrój threshold, żeby zbalansować FP i FN.",
        )

    return (
        "The background shows probability of class_1 before thresholding.",
        "Train the model, then tune threshold to balance FP and FN.",
    )


def _model_line(snapshot: AlgorithmSnapshot, *, language: str) -> str:
    """Build a compact explanation of current model quality."""
    loss = float(snapshot.metrics["loss"])
    accuracy = float(snapshot.metrics["accuracy"])
    precision = float(snapshot.metrics["precision"])
    recall = float(snapshot.metrics["recall"])

    line = (
        f"loss={loss:.4f}, accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}."
    )

    if language == "pl":
        return f"Aktualnie: {line}"

    return (
        f"Loss={loss:.4f}, accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}."
    )


def _challenge_line(
    snapshot: AlgorithmSnapshot,
    challenge_result: PrecisionRecallChallengeResult,
    *,
    language: str,
) -> str:
    """Build a compact explanation of threshold and challenge state."""
    threshold = float(snapshot.metrics["threshold"])
    false_positive = int(snapshot.metrics["false_positive"])
    false_negative = int(snapshot.metrics["false_negative"])

    if challenge_result.success:
        if language == "pl":
            return (
                f"Cel osiągnięty: threshold={threshold:.2f}, "
                f"FP={false_positive}, FN={false_negative}."
            )

        return (
            f"Challenge success: threshold={threshold:.2f}, "
            f"FP={false_positive}, FN={false_negative}."
        )

    if language == "pl":
        return (
            f"Jeszcze nie: threshold={threshold:.2f}, "
            f"FP={false_positive}, FN={false_negative}; dostrój threshold."
        )

    return (
        f"Challenge not completed: threshold={threshold:.2f}, "
        f"FP={false_positive}, FN={false_negative}; tune threshold."
    )


def _normalize_language(language: str) -> str:
    """Return a supported UI language code."""
    if language.lower().startswith("pl"):
        return "pl"

    return "en"
