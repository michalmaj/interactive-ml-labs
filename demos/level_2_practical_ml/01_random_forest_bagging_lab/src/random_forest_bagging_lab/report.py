"""CLI comparison reports for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

MODEL_SINGLE_TREE: Final[str] = "single_tree"
MODEL_RANDOM_FOREST: Final[str] = "random_forest"
MODEL_TIE: Final[str] = "tie"

METRIC_TRAIN_ACCURACY: Final[str] = "train_accuracy"
METRIC_TEST_ACCURACY: Final[str] = "test_accuracy"
METRIC_MAX_DEPTH: Final[str] = "max_depth"
METRIC_NODE_COUNT: Final[str] = "node_count"
METRIC_LEAF_COUNT: Final[str] = "leaf_count"
METRIC_TREE_COUNT: Final[str] = "tree_count"
METRIC_MEAN_TEST_CONFIDENCE: Final[str] = "mean_test_confidence"

NO_FOREST_CONFIDENCE_PLACEHOLDER: Final[str] = "n/a"


@dataclass(frozen=True, slots=True)
class ModelReportMetrics:
    """Metrics for one model in the CLI comparison report.

    Attributes:
        name: Model name.
        train_accuracy: Accuracy on the training split.
        test_accuracy: Accuracy on the test split.
        generalization_gap: Difference between train and test accuracy.
        max_depth: Configured maximum tree depth.
        node_count: Number of nodes for a single tree, if available.
        leaf_count: Number of leaves for a single tree, if available.
        tree_count: Number of trees for an ensemble, if available.
        mean_test_confidence: Mean vote confidence on the test split, if available.
    """

    name: str
    train_accuracy: float
    test_accuracy: float
    generalization_gap: float
    max_depth: int
    node_count: int | None = None
    leaf_count: int | None = None
    tree_count: int | None = None
    mean_test_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    """Comparison between a single-tree baseline and a random forest.

    Attributes:
        single_tree: Metrics for the single-tree baseline.
        forest: Metrics for the random forest.
        train_accuracy_delta: Forest train accuracy minus single-tree train accuracy.
        test_accuracy_delta: Forest test accuracy minus single-tree test accuracy.
        winner: Name of the model with higher test accuracy.
        summary: Short human-readable summary.
    """

    single_tree: ModelReportMetrics
    forest: ModelReportMetrics
    train_accuracy_delta: float
    test_accuracy_delta: float
    winner: str
    summary: str


def build_model_comparison_report(
    *,
    single_tree_snapshot: AlgorithmSnapshot,
    forest_snapshot: AlgorithmSnapshot,
) -> ModelComparisonReport:
    """Build a comparison report from fitted model snapshots.

    Args:
        single_tree_snapshot: Snapshot returned by `SingleTreeBaseline`.
        forest_snapshot: Snapshot returned by `RandomForestModel`.

    Returns:
        ModelComparisonReport with accuracy deltas and summary.
    """
    single_tree = _build_single_tree_metrics(single_tree_snapshot)
    forest = _build_forest_metrics(forest_snapshot)

    train_accuracy_delta = forest.train_accuracy - single_tree.train_accuracy
    test_accuracy_delta = forest.test_accuracy - single_tree.test_accuracy
    winner = _resolve_winner(
        single_tree_test_accuracy=single_tree.test_accuracy,
        forest_test_accuracy=forest.test_accuracy,
    )
    summary = _build_summary(
        winner=winner,
        test_accuracy_delta=test_accuracy_delta,
    )

    return ModelComparisonReport(
        single_tree=single_tree,
        forest=forest,
        train_accuracy_delta=train_accuracy_delta,
        test_accuracy_delta=test_accuracy_delta,
        winner=winner,
        summary=summary,
    )


def format_model_comparison_report(
    *,
    label: str,
    report: ModelComparisonReport,
) -> tuple[str, ...]:
    """Format a comparison report as CLI lines.

    Args:
        label: Dataset label.
        report: Comparison report.

    Returns:
        Tuple of CLI-ready lines.
    """
    return (
        f"{label} model comparison:",
        _format_model_line(report.single_tree),
        _format_model_line(report.forest),
        f"{label} train accuracy delta forest-baseline: {report.train_accuracy_delta:+.3f}",
        f"{label} test accuracy delta forest-baseline: {report.test_accuracy_delta:+.3f}",
        f"{label} winner by test accuracy: {report.winner}",
        f"{label} summary: {report.summary}",
    )


def _build_single_tree_metrics(snapshot: AlgorithmSnapshot) -> ModelReportMetrics:
    """Build report metrics for a single-tree baseline."""
    train_accuracy = _metric_float(snapshot, METRIC_TRAIN_ACCURACY)
    test_accuracy = _metric_float(snapshot, METRIC_TEST_ACCURACY)

    return ModelReportMetrics(
        name=MODEL_SINGLE_TREE,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        generalization_gap=train_accuracy - test_accuracy,
        max_depth=_metric_int(snapshot, METRIC_MAX_DEPTH),
        node_count=_optional_metric_int(snapshot, METRIC_NODE_COUNT),
        leaf_count=_optional_metric_int(snapshot, METRIC_LEAF_COUNT),
    )


def _build_forest_metrics(snapshot: AlgorithmSnapshot) -> ModelReportMetrics:
    """Build report metrics for a random forest."""
    train_accuracy = _metric_float(snapshot, METRIC_TRAIN_ACCURACY)
    test_accuracy = _metric_float(snapshot, METRIC_TEST_ACCURACY)

    return ModelReportMetrics(
        name=MODEL_RANDOM_FOREST,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        generalization_gap=train_accuracy - test_accuracy,
        max_depth=_metric_int(snapshot, METRIC_MAX_DEPTH),
        tree_count=_metric_int(snapshot, METRIC_TREE_COUNT),
        mean_test_confidence=_metric_float(snapshot, METRIC_MEAN_TEST_CONFIDENCE),
    )


def _resolve_winner(
    *,
    single_tree_test_accuracy: float,
    forest_test_accuracy: float,
) -> str:
    """Resolve winner by test accuracy."""
    if forest_test_accuracy > single_tree_test_accuracy:
        return MODEL_RANDOM_FOREST

    if single_tree_test_accuracy > forest_test_accuracy:
        return MODEL_SINGLE_TREE

    return MODEL_TIE


def _build_summary(*, winner: str, test_accuracy_delta: float) -> str:
    """Build a short interpretation of the comparison."""
    if winner == MODEL_RANDOM_FOREST:
        return (
            "Random forest has higher test accuracy than the single-tree baseline "
            f"by {test_accuracy_delta:.3f}."
        )

    if winner == MODEL_SINGLE_TREE:
        return (
            "Single tree has higher test accuracy than the random forest "
            f"by {abs(test_accuracy_delta):.3f}."
        )

    return "Single tree and random forest have the same test accuracy."


def _format_model_line(metrics: ModelReportMetrics) -> str:
    """Format one model row as a readable CLI line."""
    details = [
        f"train={metrics.train_accuracy:.3f}",
        f"test={metrics.test_accuracy:.3f}",
        f"gap={metrics.generalization_gap:.3f}",
        f"max_depth={metrics.max_depth}",
    ]

    if metrics.node_count is not None:
        details.append(f"nodes={metrics.node_count}")

    if metrics.leaf_count is not None:
        details.append(f"leaves={metrics.leaf_count}")

    if metrics.tree_count is not None:
        details.append(f"trees={metrics.tree_count}")

    if metrics.mean_test_confidence is not None:
        details.append(f"mean_confidence={metrics.mean_test_confidence:.3f}")
    else:
        details.append(f"mean_confidence={NO_FOREST_CONFIDENCE_PLACEHOLDER}")

    return f"  {metrics.name}: " + ", ".join(details)


def _metric_float(snapshot: AlgorithmSnapshot, key: str) -> float:
    """Read a required metric as float."""
    try:
        return float(snapshot.metrics[key])
    except KeyError as error:
        msg = f"Snapshot metric {key!r} is required."
        raise ValueError(msg) from error


def _metric_int(snapshot: AlgorithmSnapshot, key: str) -> int:
    """Read a required metric as int."""
    try:
        return int(snapshot.metrics[key])
    except KeyError as error:
        msg = f"Snapshot metric {key!r} is required."
        raise ValueError(msg) from error


def _optional_metric_int(snapshot: AlgorithmSnapshot, key: str) -> int | None:
    """Read an optional metric as int."""
    if key not in snapshot.metrics:
        return None

    return int(snapshot.metrics[key])
