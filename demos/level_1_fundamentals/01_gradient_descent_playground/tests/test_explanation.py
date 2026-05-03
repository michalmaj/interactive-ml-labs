"""Tests for Gradient Descent Playground explanation helpers."""

from gradient_descent_playground.challenge import ChallengeResult
from gradient_descent_playground.explanation import build_explanation_lines
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    iteration: int,
    loss: float = 2.0,
    annotations: tuple[str, ...] = (),
) -> AlgorithmSnapshot:
    """Create a minimal snapshot for explanation tests."""
    return AlgorithmSnapshot(
        iteration=iteration,
        metrics={
            "loss": loss,
            "weight": 1.25,
            "bias": -0.5,
        },
        annotations=annotations,
    )


def _challenge_result(*, status: str) -> ChallengeResult:
    """Create a minimal challenge result for explanation tests."""
    return ChallengeResult(
        status=status,
        target_loss=1.0,
        current_loss=2.0,
        current_step=3,
        max_steps=10,
        steps_remaining=7,
        message="Challenge message.",
    )


def test_explanation_describes_initial_state() -> None:
    """Initial state should explain how to start stepping."""
    lines = build_explanation_lines(
        _snapshot(iteration=0),
        _challenge_result(status="active"),
    )

    assert lines == (
        "The model starts with initial weight and bias.",
        "Press N for one step or Space to run gradient descent.",
    )


def test_explanation_uses_algorithm_annotations() -> None:
    """Algorithm annotations should be shown when available."""
    lines = build_explanation_lines(
        _snapshot(
            iteration=1,
            annotations=(
                "Updated weight by 0.123456.",
                "Updated bias by -0.012345.",
            ),
        ),
        _challenge_result(status="active"),
    )

    assert lines == (
        "Updated weight by 0.123456.",
        "Updated bias by -0.012345.",
    )


def test_explanation_handles_successful_challenge() -> None:
    """Successful challenge should override normal algorithm explanation."""
    lines = build_explanation_lines(
        _snapshot(iteration=10, loss=0.8),
        _challenge_result(status="success"),
    )

    assert lines[0] == "Challenge completed: the model reached the target loss."


def test_explanation_handles_failed_challenge() -> None:
    """Failed challenge should override normal algorithm explanation."""
    lines = build_explanation_lines(
        _snapshot(iteration=10, loss=2.0),
        _challenge_result(status="failed"),
    )

    assert lines[0] == "Challenge failed: the step limit was reached before target loss."
