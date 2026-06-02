"""Explanation helpers for the Decision Tree Splitter demo."""

from __future__ import annotations

from typing import Final

from ml_lab_core import AlgorithmSnapshot

from decision_tree_splitter.challenge import DecisionTreeChallengeResult

MODE_AUTO_TREE: Final[str] = "auto_tree"
MODE_MANUAL_SPLIT: Final[str] = "manual_split"


def build_explanation_text(
    snapshot: AlgorithmSnapshot,
    *,
    mode: str,
    manual_error: str | None,
    challenge_result: DecisionTreeChallengeResult,
    language: str = "en",
) -> str:
    """Build a short student-facing explanation for the current UI state.

    Args:
        snapshot: Current recursive decision tree snapshot.
        mode: Current UI mode.
        manual_error: Manual split error if selected split is invalid.
        challenge_result: Current challenge result.

    Returns:
        Explanation text ready to be displayed in the bottom panel.
    """
    language = _normalize_language(language)

    if mode == MODE_MANUAL_SPLIT:
        return _manual_mode_explanation(manual_error, language=language)

    if challenge_result.success:
        if language == "pl":
            return "Cel osiągnięty: drzewo ma dobrą accuracy i mieści się w limicie depth."

        return challenge_result.message

    return _auto_mode_explanation(snapshot, challenge_result, language=language)


def _manual_mode_explanation(manual_error: str | None, *, language: str) -> str:
    """Build explanation for manual split mode."""
    if manual_error is not None:
        if language == "pl":
            return "Manual split jest niepoprawny, bo nie tworzy dwóch użytecznych gałęzi."

        return "Manual split is invalid because it does not create two useful children."

    if language == "pl":
        return "Manual mode: przesuń split i porównaj impurity/gain z własną intuicją."

    return "Manual mode: move the split and compare impurity/gain with your intuition."


def _auto_mode_explanation(
    snapshot: AlgorithmSnapshot,
    challenge_result: DecisionTreeChallengeResult,
    *,
    language: str,
) -> str:
    """Build explanation for automatic recursive tree mode."""
    accuracy = float(snapshot.metrics["training_accuracy"])
    max_depth = int(snapshot.metrics["max_depth"])
    leaf_count = int(snapshot.metrics["leaf_count"])

    if language == "pl":
        return (
            "Auto mode: rekurencyjne axis-aligned splity. "
            f"max_depth={max_depth}, leaves={leaf_count}, accuracy={accuracy:.2f}. "
            f"{_status_text(challenge_result.status)}: "
            f"target accuracy={challenge_result.target_accuracy:.2f}."
        )

    return (
        f"Auto mode: recursive axis-aligned splits. "
        f"max_depth={max_depth}, leaves={leaf_count}, accuracy={accuracy:.2f}. "
        f"{challenge_result.status}: target accuracy={challenge_result.target_accuracy:.2f}."
    )


def _normalize_language(language: str) -> str:
    """Return a supported UI language code."""
    if language.lower().startswith("pl"):
        return "pl"

    return "en"


def _status_text(status: str) -> str:
    """Return Polish compact status text."""
    return {"success": "gotowe", "failed": "do poprawy"}.get(status, status)
