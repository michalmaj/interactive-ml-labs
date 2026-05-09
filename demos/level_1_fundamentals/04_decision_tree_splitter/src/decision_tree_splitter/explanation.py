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
    if mode == MODE_MANUAL_SPLIT:
        return _manual_mode_explanation(manual_error)

    if challenge_result.success:
        return challenge_result.message

    return _auto_mode_explanation(snapshot, challenge_result)


def _manual_mode_explanation(manual_error: str | None) -> str:
    """Build explanation for manual split mode."""
    if manual_error is not None:
        return "Manual split is invalid because it does not create two useful children."

    return "Manual mode: move the split and compare impurity/gain with your intuition."


def _auto_mode_explanation(
    snapshot: AlgorithmSnapshot,
    challenge_result: DecisionTreeChallengeResult,
) -> str:
    """Build explanation for automatic recursive tree mode."""
    accuracy = float(snapshot.metrics["training_accuracy"])
    max_depth = int(snapshot.metrics["max_depth"])
    leaf_count = int(snapshot.metrics["leaf_count"])

    return (
        f"Auto mode: recursive axis-aligned splits. "
        f"max_depth={max_depth}, leaves={leaf_count}, accuracy={accuracy:.2f}. "
        f"{challenge_result.status}: target accuracy={challenge_result.target_accuracy:.2f}."
    )
