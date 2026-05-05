"""Explanation helpers for the k-NN Vote Map demo."""

from __future__ import annotations

from typing import Final

from ml_lab_core import AlgorithmSnapshot

from knn_vote_map.challenge import KNNAccuracyChallengeResult

MAX_EXPLANATION_LINES: Final[int] = 2


def build_explanation_lines(
    snapshot: AlgorithmSnapshot,
    challenge_result: KNNAccuracyChallengeResult,
) -> tuple[str, ...]:
    """Build short student-facing explanation lines.

    Args:
        snapshot: Current classifier snapshot.
        challenge_result: Current challenge result.

    Returns:
        Short explanation lines ready to be displayed by the renderer.
    """
    if not snapshot.metrics.get("has_prediction"):
        return _initial_lines(challenge_result)

    predicted_label = int(snapshot.metrics["predicted_label"])
    k = int(snapshot.metrics["k"])
    vote_counts = snapshot.visual_state.get("vote_counts", {})

    vote_line = _vote_line(
        predicted_label=predicted_label,
        k=k,
        vote_counts=vote_counts,
    )
    challenge_line = _challenge_line(challenge_result)

    return (vote_line, challenge_line)[:MAX_EXPLANATION_LINES]


def _initial_lines(challenge_result: KNNAccuracyChallengeResult) -> tuple[str, ...]:
    """Build explanation lines before the first query point is classified."""
    if challenge_result.success:
        return (
            "Challenge completed on hidden test data.",
            "Click the map or press N to inspect local k-NN decisions.",
        )

    return (
        "Challenge is not completed on hidden test data yet.",
        "Try changing k, noise, or seed; click the map to inspect decisions.",
    )


def _vote_line(
    *,
    predicted_label: int,
    k: int,
    vote_counts: object,
) -> str:
    """Build a compact explanation of the voting result."""
    if not isinstance(vote_counts, dict) or not vote_counts:
        return f"The query point was classified as class_{predicted_label} using k={k}."

    sorted_votes = sorted((int(label), int(count)) for label, count in vote_counts.items())
    votes_text = ", ".join(f"class_{label}: {count}" for label, count in sorted_votes)

    return f"Prediction: class_{predicted_label}. Neighbor votes: {votes_text}."


def _challenge_line(challenge_result: KNNAccuracyChallengeResult) -> str:
    """Build a compact explanation of the challenge state."""
    if challenge_result.success:
        return (
            f"Challenge success: test accuracy is {challenge_result.accuracy:.2f} "
            f"with target {challenge_result.target_accuracy:.2f}."
        )

    return (
        f"Challenge not completed: accuracy is {challenge_result.accuracy:.2f}; "
        "try tuning k or reducing noise."
    )