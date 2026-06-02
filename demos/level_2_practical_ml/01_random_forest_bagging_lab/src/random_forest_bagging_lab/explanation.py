"""Explanation helpers for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from random_forest_bagging_lab.challenge import RandomForestChallengeResult


def build_bottom_panel_explanation(
    *,
    confidence_view_enabled: bool,
    challenge_result: RandomForestChallengeResult,
    language: str = "en",
) -> str:
    """Build bottom-panel explanation text for the Pygame UI.

    Args:
        confidence_view_enabled: Whether confidence view is enabled.
        challenge_result: Current challenge result.

    Returns:
        Student-facing explanation text.
    """
    language = _normalize_language(language)

    if challenge_result.success:
        if language == "pl":
            return "Cel osiągnięty: forest dobrze generalizuje bez zbyt wielu drzew."

        return challenge_result.message

    confidence_text = build_confidence_view_explanation(
        confidence_view_enabled=confidence_view_enabled,
        language=language,
    )

    if language == "pl":
        return f"{_challenge_message(challenge_result)} {confidence_text}"

    return f"{challenge_result.message} {confidence_text}"


def build_confidence_view_explanation(
    *,
    confidence_view_enabled: bool,
    language: str = "en",
) -> str:
    """Build short explanation for confidence view state.

    Args:
        confidence_view_enabled: Whether confidence view is enabled.

    Returns:
        Explanation text.
    """
    language = _normalize_language(language)

    if language == "pl":
        if confidence_view_enabled:
            return "Confidence view: jaśniejsze regiony forest oznaczają słabszą zgodność drzew."

        return "Confidence view jest wyłączony: regiony forest pokazują tylko finalną klasę."

    if confidence_view_enabled:
        return "Confidence view: pale forest regions mean weaker agreement between trees."

    return "Confidence view is off: forest regions show final class only."


def build_challenge_target_text(challenge_result: RandomForestChallengeResult) -> str:
    """Build compact challenge target progress text.

    Args:
        challenge_result: Current challenge result.

    Returns:
        Compact target text.
    """
    return (
        f"{challenge_result.forest_test_accuracy:.2f}/{challenge_result.target_test_accuracy:.2f}"
    )


def build_tree_limit_text(challenge_result: RandomForestChallengeResult) -> str:
    """Build compact tree-limit progress text.

    Args:
        challenge_result: Current challenge result.

    Returns:
        Compact tree-limit text.
    """
    return f"{challenge_result.forest_tree_count}/{challenge_result.max_tree_count}"


def build_gap_limit_text(challenge_result: RandomForestChallengeResult) -> str:
    """Build compact generalization-gap progress text.

    Args:
        challenge_result: Current challenge result.

    Returns:
        Compact gap-limit text.
    """
    return (
        f"{challenge_result.forest_generalization_gap:.2f}/"
        f"{challenge_result.max_generalization_gap:.2f}"
    )


def _normalize_language(language: str) -> str:
    """Return a supported UI language code."""
    if language.lower().startswith("pl"):
        return "pl"

    return "en"


def _challenge_message(challenge_result: RandomForestChallengeResult) -> str:
    """Build a natural Polish challenge message."""
    failed: list[str] = []

    if challenge_result.forest_test_accuracy < challenge_result.target_test_accuracy:
        failed.append("test accuracy jest za niskie")
    if challenge_result.forest_tree_count > challenge_result.max_tree_count:
        failed.append("forest używa za wielu drzew")
    if challenge_result.forest_generalization_gap > challenge_result.max_generalization_gap:
        failed.append("generalization gap jest za duży")

    if not failed:
        return "Cel jest blisko, ale konfiguracja wymaga jeszcze dostrojenia."

    return "Cel nieosiągnięty: " + ", ".join(failed) + "."
