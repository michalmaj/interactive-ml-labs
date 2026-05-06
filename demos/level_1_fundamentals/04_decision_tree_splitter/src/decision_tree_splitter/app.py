"""Command-line entry point for the Decision Tree Splitter demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

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
from decision_tree_splitter.split import best_split


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive implementation will be added in later pull requests.
    This entry point verifies that the package can generate datasets, compute
    impurity metrics, and find the best first split.
    """
    axis_dataset = make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    axis_features = np.asarray(axis_dataset.features, dtype=float)
    axis_targets = np.asarray(axis_dataset.targets, dtype=int)
    xor_features = np.asarray(xor_dataset.features, dtype=float)
    xor_targets = np.asarray(xor_dataset.targets, dtype=int)

    axis_class_counts = np.bincount(axis_targets, minlength=2)
    xor_class_counts = np.bincount(xor_targets, minlength=2)

    axis_best_split = best_split(axis_features, axis_targets)
    xor_best_split = best_split(xor_features, xor_targets)

    history = MetricsHistory()
    history.add("axis_sample_count", float(axis_features.shape[0]))
    history.add("xor_sample_count", float(xor_features.shape[0]))
    history.add("axis_class_0_count", float(axis_class_counts[0]))
    history.add("axis_class_1_count", float(axis_class_counts[1]))
    history.add("xor_class_0_count", float(xor_class_counts[0]))
    history.add("xor_class_1_count", float(xor_class_counts[1]))
    history.add("axis_gini", gini_impurity(axis_targets))
    history.add("axis_entropy", entropy_impurity(axis_targets))
    history.add("xor_gini", gini_impurity(xor_targets))
    history.add("xor_entropy", entropy_impurity(xor_targets))
    history.add("axis_best_gain", axis_best_split.information_gain)
    history.add("xor_best_gain", xor_best_split.information_gain)

    print("Decision Tree Splitter")
    print("Synthetic classification datasets generated successfully.")

    print(f"Axis-aligned features shape: {axis_features.shape}")
    print(f"Axis-aligned targets shape: {axis_targets.shape}")
    print(f"Axis-aligned class 0 count: {history.latest('axis_class_0_count'):.0f}")
    print(f"Axis-aligned class 1 count: {history.latest('axis_class_1_count'):.0f}")
    print(f"Axis-aligned root Gini: {history.latest('axis_gini'):.3f}")
    print(f"Axis-aligned root entropy: {history.latest('axis_entropy'):.3f}")
    print(
        "Axis-aligned best split: "
        f"x{axis_best_split.candidate.feature_index + 1} <= "
        f"{axis_best_split.candidate.threshold:.3f}, "
        f"gain={history.latest('axis_best_gain'):.3f}",
    )

    print(f"XOR features shape: {xor_features.shape}")
    print(f"XOR targets shape: {xor_targets.shape}")
    print(f"XOR class 0 count: {history.latest('xor_class_0_count'):.0f}")
    print(f"XOR class 1 count: {history.latest('xor_class_1_count'):.0f}")
    print(f"XOR root Gini: {history.latest('xor_gini'):.3f}")
    print(f"XOR root entropy: {history.latest('xor_entropy'):.3f}")
    print(
        "XOR best split: "
        f"x{xor_best_split.candidate.feature_index + 1} <= "
        f"{xor_best_split.candidate.threshold:.3f}, "
        f"gain={history.latest('xor_best_gain'):.3f}",
    )
