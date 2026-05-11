"""Explanation helpers for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from random_forest_bagging_lab.challenge import RandomForestChallengeResult


def build_bottom_panel_explanation(
    *,
    confidence_view_enabled: bool,
    challenge_result: RandomForestChallengeResult,
) -> str:
    """Build bottom-panel explanation text for the Pygame UI.

    Args:
        confidence_view_enabled: Whether confidence view is enabled.
        challenge_result: Current challenge result.

    Returns:
        Student-facing explanation text.
    """
    if challenge_result.success:
        return challenge_result.message

    confidence_text = build_confidence_view_explanation(
        confidence_view_enabled=confidence_view_enabled,
    )

    return f"{challenge_result.message} {confidence_text}"


def build_confidence_view_explanation(*, confidence_view_enabled: bool) -> str:
    """Build short explanation for confidence view state.

    Args:
        confidence_view_enabled: Whether confidence view is enabled.

    Returns:
        Explanation text.
    """
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
