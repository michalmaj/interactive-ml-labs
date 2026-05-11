"""Tests for Random Forest explanation helpers."""

from random_forest_bagging_lab import (
    RandomForestChallengeResult,
    build_bottom_panel_explanation,
    build_challenge_target_text,
    build_confidence_view_explanation,
    build_gap_limit_text,
    build_tree_limit_text,
)


def _challenge_result(*, success: bool) -> RandomForestChallengeResult:
    """Create a minimal challenge result for explanation tests."""
    return RandomForestChallengeResult(
        status="success" if success else "failed",
        target_test_accuracy=0.90,
        max_tree_count=25,
        max_generalization_gap=0.15,
        forest_test_accuracy=0.92 if success else 0.70,
        forest_train_accuracy=0.95,
        forest_tree_count=25,
        forest_generalization_gap=0.03 if success else 0.25,
        baseline_test_accuracy=0.80,
        test_accuracy_delta=0.12 if success else -0.10,
        winner="random_forest",
        message=(
            "Challenge completed: forest generalizes well without using too many trees."
            if success
            else "Challenge not completed: test accuracy is too low."
        ),
    )


def test_build_bottom_panel_explanation_returns_success_message() -> None:
    """Successful challenge should show challenge completion message."""
    text = build_bottom_panel_explanation(
        confidence_view_enabled=False,
        challenge_result=_challenge_result(success=True),
    )

    assert text == "Challenge completed: forest generalizes well without using too many trees."


def test_build_bottom_panel_explanation_includes_confidence_message_when_failed() -> None:
    """Failed challenge should include confidence-view explanation."""
    text = build_bottom_panel_explanation(
        confidence_view_enabled=True,
        challenge_result=_challenge_result(success=False),
    )

    assert text == (
        "Challenge not completed: test accuracy is too low. "
        "Confidence view: pale forest regions mean weaker agreement between trees."
    )


def test_build_confidence_view_explanation_when_enabled() -> None:
    """Enabled confidence view should explain pale regions."""
    text = build_confidence_view_explanation(confidence_view_enabled=True)

    assert text == "Confidence view: pale forest regions mean weaker agreement between trees."


def test_build_confidence_view_explanation_when_disabled() -> None:
    """Disabled confidence view should explain final-class-only coloring."""
    text = build_confidence_view_explanation(confidence_view_enabled=False)

    assert text == "Confidence view is off: forest regions show final class only."


def test_build_challenge_target_text() -> None:
    """Target text should show current test accuracy and required target."""
    text = build_challenge_target_text(_challenge_result(success=True))

    assert text == "0.92/0.90"


def test_build_tree_limit_text() -> None:
    """Tree-limit text should show current tree count and maximum allowed count."""
    text = build_tree_limit_text(_challenge_result(success=True))

    assert text == "25/25"


def test_build_gap_limit_text() -> None:
    """Gap-limit text should show current gap and allowed maximum gap."""
    text = build_gap_limit_text(_challenge_result(success=True))

    assert text == "0.03/0.15"
