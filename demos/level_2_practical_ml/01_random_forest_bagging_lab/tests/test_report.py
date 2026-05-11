"""Tests for CLI comparison reports."""

import pytest
from ml_lab_core import AlgorithmSnapshot
from random_forest_bagging_lab import (
    ModelComparisonReport,
    ModelReportMetrics,
    build_model_comparison_report,
    format_model_comparison_report,
)

SINGLE_TREE_TRAIN_ACCURACY: float = 0.95
SINGLE_TREE_TEST_ACCURACY: float = 0.80
FOREST_TRAIN_ACCURACY: float = 0.93
FOREST_TEST_ACCURACY: float = 0.90
FOREST_MEAN_TEST_CONFIDENCE: float = 0.86
EXPECTED_TEST_DELTA: float = FOREST_TEST_ACCURACY - SINGLE_TREE_TEST_ACCURACY
EXPECTED_TRAIN_DELTA: float = FOREST_TRAIN_ACCURACY - SINGLE_TREE_TRAIN_ACCURACY


def _single_tree_snapshot(
    *,
    train_accuracy: float = SINGLE_TREE_TRAIN_ACCURACY,
    test_accuracy: float = SINGLE_TREE_TEST_ACCURACY,
) -> AlgorithmSnapshot:
    """Create a minimal single-tree baseline snapshot."""
    return AlgorithmSnapshot(
        iteration=3,
        status="fitted",
        metrics={
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "max_depth": 2,
            "node_count": 7,
            "leaf_count": 4,
        },
        visual_state={},
        done=True,
    )


def _forest_snapshot(
    *,
    train_accuracy: float = FOREST_TRAIN_ACCURACY,
    test_accuracy: float = FOREST_TEST_ACCURACY,
) -> AlgorithmSnapshot:
    """Create a minimal random forest snapshot."""
    return AlgorithmSnapshot(
        iteration=25,
        status="fitted",
        metrics={
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "max_depth": 2,
            "tree_count": 25,
            "mean_test_confidence": FOREST_MEAN_TEST_CONFIDENCE,
        },
        visual_state={},
        done=True,
    )


def test_build_model_comparison_report_returns_report() -> None:
    """Comparison builder should return a report object."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(),
        forest_snapshot=_forest_snapshot(),
    )

    assert isinstance(report, ModelComparisonReport)
    assert isinstance(report.single_tree, ModelReportMetrics)
    assert isinstance(report.forest, ModelReportMetrics)


def test_build_model_comparison_report_reads_single_tree_metrics() -> None:
    """Single-tree metrics should be read from the snapshot."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(),
        forest_snapshot=_forest_snapshot(),
    )

    assert report.single_tree.name == "single_tree"
    assert report.single_tree.train_accuracy == pytest.approx(SINGLE_TREE_TRAIN_ACCURACY)
    assert report.single_tree.test_accuracy == pytest.approx(SINGLE_TREE_TEST_ACCURACY)
    assert report.single_tree.generalization_gap == pytest.approx(0.15)
    assert report.single_tree.max_depth == 2
    assert report.single_tree.node_count == 7
    assert report.single_tree.leaf_count == 4
    assert report.single_tree.tree_count is None
    assert report.single_tree.mean_test_confidence is None


def test_build_model_comparison_report_reads_forest_metrics() -> None:
    """Forest metrics should be read from the snapshot."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(),
        forest_snapshot=_forest_snapshot(),
    )

    assert report.forest.name == "random_forest"
    assert report.forest.train_accuracy == pytest.approx(FOREST_TRAIN_ACCURACY)
    assert report.forest.test_accuracy == pytest.approx(FOREST_TEST_ACCURACY)
    assert report.forest.generalization_gap == pytest.approx(0.03)
    assert report.forest.max_depth == 2
    assert report.forest.tree_count == 25
    assert report.forest.mean_test_confidence == pytest.approx(FOREST_MEAN_TEST_CONFIDENCE)


def test_build_model_comparison_report_computes_accuracy_deltas() -> None:
    """Accuracy deltas should be forest minus single-tree metrics."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(),
        forest_snapshot=_forest_snapshot(),
    )

    assert report.train_accuracy_delta == pytest.approx(EXPECTED_TRAIN_DELTA)
    assert report.test_accuracy_delta == pytest.approx(EXPECTED_TEST_DELTA)


def test_build_model_comparison_report_selects_forest_when_test_accuracy_is_higher() -> None:
    """Forest should win when it has higher test accuracy."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(test_accuracy=0.80),
        forest_snapshot=_forest_snapshot(test_accuracy=0.90),
    )

    assert report.winner == "random_forest"
    assert "Random forest has higher test accuracy" in report.summary


def test_build_model_comparison_report_selects_single_tree_when_test_accuracy_is_higher() -> None:
    """Single tree should win when it has higher test accuracy."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(test_accuracy=0.95),
        forest_snapshot=_forest_snapshot(test_accuracy=0.90),
    )

    assert report.winner == "single_tree"
    assert "Single tree has higher test accuracy" in report.summary


def test_build_model_comparison_report_reports_tie() -> None:
    """Report should explicitly handle equal test accuracy."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(test_accuracy=0.90),
        forest_snapshot=_forest_snapshot(test_accuracy=0.90),
    )

    assert report.winner == "tie"
    assert report.summary == "Single tree and random forest have the same test accuracy."


def test_format_model_comparison_report_returns_cli_lines() -> None:
    """Formatted report should contain readable CLI lines."""
    report = build_model_comparison_report(
        single_tree_snapshot=_single_tree_snapshot(),
        forest_snapshot=_forest_snapshot(),
    )

    lines = format_model_comparison_report(
        label="XOR",
        report=report,
    )

    assert lines[0] == "XOR model comparison:"
    assert "single_tree" in lines[1]
    assert "random_forest" in lines[2]
    assert "train accuracy delta" in lines[3]
    assert "test accuracy delta" in lines[4]
    assert "winner by test accuracy" in lines[5]
    assert "summary" in lines[6]


def test_build_model_comparison_report_rejects_missing_metrics() -> None:
    """Missing required metrics should fail clearly."""
    broken_snapshot = AlgorithmSnapshot(
        iteration=1,
        status="fitted",
        metrics={
            "train_accuracy": 1.0,
        },
        visual_state={},
        done=True,
    )

    with pytest.raises(ValueError, match="required"):
        build_model_comparison_report(
            single_tree_snapshot=broken_snapshot,
            forest_snapshot=_forest_snapshot(),
        )
