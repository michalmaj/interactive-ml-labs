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
    if challenge_result.success:
        return (
            "Challenge completed: the model reached the target loss.",
            "Try increasing noise or lowering learning rate to make it harder.",
        )

    if challenge_result.failed:
        return (
            "Challenge failed: the step limit was reached before target loss.",
            "Try increasing learning rate carefully or reducing data noise.",
        )

    if snapshot.iteration == INITIAL_STEP:
        return (
            "The model starts with initial weight and bias.",
            "Press N for one step or Space to run gradient descent.",
        )

    annotation_lines = _annotation_lines(snapshot)

    if annotation_lines:
        return annotation_lines

    return (
        "The model is updating parameters to reduce mean squared error.",
        _current_state_line(snapshot),
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


def _current_state_line(snapshot: AlgorithmSnapshot) -> str:
    """Build a compact line with current loss and parameters."""
    loss = float(snapshot.metrics["loss"])
    weight = float(snapshot.metrics["weight"])
    bias = float(snapshot.metrics["bias"])

    return f"Current loss: {loss:.4f}, weight: {weight:.3f}, bias: {bias:.3f}."
