"""Tests for Boosting Mistake Lab explanation messages."""

import pytest
from boosting_mistake_lab import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    BoostingExplanation,
    build_boosting_explanation,
)
from ml_lab_core import AlgorithmSnapshot


def _snapshot(
    *,
    boosted_test_accuracy: float,
    completed_round_count: int,
    boosted_generalization_gap: float,
    best_staged_round_index: int,
    best_staged_boosted_test_accuracy: float,
) -> AlgorithmSnapshot:
    """Create a minimal trainer snapshot for explanation tests."""
    return AlgorithmSnapshot(
        iteration=completed_round_count,
        status="completed",
        metrics={
            "boosted_test_accuracy": boosted_test_accuracy,
            "completed_round_count": completed_round_count,
            "boosted_generalization_gap": boosted_generalization_gap,
            "best_staged_round_index": best_staged_round_index,
            "best_staged_boosted_test_accuracy": best_staged_boosted_test_accuracy,
        },
        visual_state={},
        annotations=(),
        done=True,
    )


def test_build_boosting_explanation_returns_explanation() -> None:
    """Explanation builder should return an explanation object."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=4,
            best_staged_boosted_test_accuracy=0.92,
        ),
        selected_stage=4,
        confidence_view_enabled=True,
    )

    assert isinstance(explanation, BoostingExplanation)


def test_build_boosting_explanation_reports_success() -> None:
    """Explanation should report success when challenge passes."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=4,
            best_staged_boosted_test_accuracy=0.92,
        ),
        selected_stage=4,
        confidence_view_enabled=True,
    )

    assert explanation.status == STATUS_SUCCESS
    assert "passed" in explanation.title.lower()
    assert "Good balance" in explanation.summary


def test_build_boosting_explanation_reports_failure() -> None:
    """Explanation should report failure when challenge fails."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.70,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=3,
            best_staged_boosted_test_accuracy=0.78,
        ),
        selected_stage=2,
        confidence_view_enabled=True,
    )

    assert explanation.status == STATUS_FAILED
    assert "needs tuning" in explanation.title.lower()
    assert any("increasing rounds" in hint for hint in explanation.hints)


def test_build_boosting_explanation_mentions_confidence_view() -> None:
    """Explanation should mention confidence view state."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=4,
            best_staged_boosted_test_accuracy=0.92,
        ),
        selected_stage=4,
        confidence_view_enabled=True,
    )

    assert any("Confidence view is on" in message for message in explanation.messages)


def test_build_boosting_explanation_can_use_polish_copy() -> None:
    """Explanation builder should support Polish UI copy for the demo shell."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=4,
            best_staged_boosted_test_accuracy=0.92,
        ),
        selected_stage=4,
        confidence_view_enabled=True,
        language="pl",
    )

    assert explanation.title == "Cel osiągnięty"
    assert "Dobry balans" in explanation.summary
    assert any("Confidence view jest włączony" in message for message in explanation.messages)


def test_build_boosting_explanation_suggests_moving_toward_best_round() -> None:
    """Explanation should suggest moving toward the best staged round."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=4,
            best_staged_boosted_test_accuracy=0.92,
        ),
        selected_stage=2,
        confidence_view_enabled=False,
    )

    assert any("move forward" in hint for hint in explanation.hints)


def test_build_boosting_explanation_suggests_checking_drop_after_best_round() -> None:
    """Explanation should mention stages after the best test round."""
    explanation = build_boosting_explanation(
        trainer_snapshot=_snapshot(
            boosted_test_accuracy=0.90,
            completed_round_count=5,
            boosted_generalization_gap=0.10,
            best_staged_round_index=2,
            best_staged_boosted_test_accuracy=0.94,
        ),
        selected_stage=5,
        confidence_view_enabled=False,
    )

    assert any("after the best test round" in hint for hint in explanation.hints)


@pytest.mark.parametrize(
    "selected_stage, expected_message",
    [
        (0, "selected_stage"),
        (6, "selected_stage"),
    ],
)
def test_build_boosting_explanation_rejects_invalid_selected_stage(
    selected_stage: int,
    expected_message: str,
) -> None:
    """Invalid selected stage should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        build_boosting_explanation(
            trainer_snapshot=_snapshot(
                boosted_test_accuracy=0.90,
                completed_round_count=5,
                boosted_generalization_gap=0.10,
                best_staged_round_index=4,
                best_staged_boosted_test_accuracy=0.92,
            ),
            selected_stage=selected_stage,
            confidence_view_enabled=True,
        )
