"""Command-line entry point for the Decision Tree Splitter demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot, MetricsHistory

from decision_tree_splitter.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from decision_tree_splitter.impurity import (
    entropy_impurity,
    gini_impurity,
)
from decision_tree_splitter.manual_split import (
    ManualSplitConfig,
    ManualSplitPrototype,
)
from decision_tree_splitter.split import SplitEvaluation, best_split
from decision_tree_splitter.stump import DecisionStump
from decision_tree_splitter.tree import (
    DecisionTreeConfig,
    RecursiveDecisionTree,
)

DEMO_TREE_MAX_DEPTH: int = 2


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive implementation will be added in later pull requests.
    This entry point verifies that the package can generate datasets, compute
    impurity metrics, evaluate manual splits, fit a stump, and fit a recursive
    decision tree.
    """
    axis_report = _build_dataset_report(DATASET_KIND_AXIS_ALIGNED)
    xor_report = _build_dataset_report(DATASET_KIND_XOR)

    print("Decision Tree Splitter")
    print("Synthetic classification datasets generated successfully.")
    _print_dataset_report("Axis-aligned", axis_report)
    _print_dataset_report("XOR", xor_report)


def _build_dataset_report(dataset_kind: str) -> dict[str, object]:
    """Build a CLI report for one dataset variant."""
    dataset = make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(dataset_kind=dataset_kind),
    )

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    manual_snapshot = _make_manual_split_snapshot(dataset)
    stump_snapshot = _make_stump_snapshot(dataset)
    tree_snapshot = _make_tree_snapshot(dataset)
    split_evaluation = best_split(features, targets)

    history = MetricsHistory()
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))
    history.add("gini", gini_impurity(targets))
    history.add("entropy", entropy_impurity(targets))
    history.add("manual_gain", float(manual_snapshot.metrics["information_gain"]))
    history.add("best_gain", split_evaluation.information_gain)
    history.add("stump_accuracy", float(stump_snapshot.metrics["training_accuracy"]))
    history.add("tree_accuracy", float(tree_snapshot.metrics["training_accuracy"]))
    history.add("tree_node_count", float(tree_snapshot.metrics["node_count"]))
    history.add("tree_leaf_count", float(tree_snapshot.metrics["leaf_count"]))

    return {
        "features_shape": features.shape,
        "targets_shape": targets.shape,
        "class_0_count": history.latest("class_0_count"),
        "class_1_count": history.latest("class_1_count"),
        "gini": history.latest("gini"),
        "entropy": history.latest("entropy"),
        "manual_snapshot": manual_snapshot,
        "best_split": split_evaluation,
        "manual_gain": history.latest("manual_gain"),
        "best_gain": history.latest("best_gain"),
        "stump_accuracy": history.latest("stump_accuracy"),
        "tree_accuracy": history.latest("tree_accuracy"),
        "tree_node_count": history.latest("tree_node_count"),
        "tree_leaf_count": history.latest("tree_leaf_count"),
    }


def _print_dataset_report(label: str, report: dict[str, object]) -> None:
    """Print one dataset report."""
    manual_snapshot = report["manual_snapshot"]
    best = report["best_split"]

    if not isinstance(manual_snapshot, AlgorithmSnapshot):
        raise TypeError("manual_snapshot has unexpected type.")

    if not isinstance(best, SplitEvaluation):
        raise TypeError("best_split has unexpected type.")

    print(f"{label} features shape: {report['features_shape']}")
    print(f"{label} targets shape: {report['targets_shape']}")
    print(f"{label} class 0 count: {float(report['class_0_count']):.0f}")
    print(f"{label} class 1 count: {float(report['class_1_count']):.0f}")
    print(f"{label} root Gini: {float(report['gini']):.3f}")
    print(f"{label} root entropy: {float(report['entropy']):.3f}")
    print(
        f"{label} manual split: "
        f"x{int(manual_snapshot.metrics['feature_index']) + 1} <= "
        f"{float(manual_snapshot.metrics['threshold']):.3f}, "
        f"gain={float(report['manual_gain']):.3f}",
    )
    print(
        f"{label} best split: "
        f"x{best.candidate.feature_index + 1} <= "
        f"{best.candidate.threshold:.3f}, "
        f"gain={float(report['best_gain']):.3f}",
    )
    print(f"{label} stump training accuracy: {float(report['stump_accuracy']):.3f}")
    print(f"{label} tree training accuracy: {float(report['tree_accuracy']):.3f}")
    print(f"{label} tree nodes: {float(report['tree_node_count']):.0f}")
    print(f"{label} tree leaves: {float(report['tree_leaf_count']):.0f}")


def _make_manual_split_snapshot(dataset: DatasetSnapshot) -> AlgorithmSnapshot:
    """Create a manual split snapshot for CLI reporting."""
    manual = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
            criterion="gini",
        ),
    )

    return manual.reset(dataset)


def _make_stump_snapshot(dataset: DatasetSnapshot) -> AlgorithmSnapshot:
    """Create a decision stump snapshot for CLI reporting."""
    stump = DecisionStump()

    return stump.reset(dataset)


def _make_tree_snapshot(dataset: DatasetSnapshot) -> AlgorithmSnapshot:
    """Create a recursive decision tree snapshot for CLI reporting."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(
            max_depth=DEMO_TREE_MAX_DEPTH,
        ),
    )

    return tree.reset(dataset)
