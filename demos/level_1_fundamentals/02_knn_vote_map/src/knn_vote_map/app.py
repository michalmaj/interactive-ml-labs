"""Command-line entry point for the k-NN Vote Map demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from knn_vote_map.dataset import make_synthetic_classification_dataset


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point currently verifies that the package can generate a dataset
    and use shared utilities from `ml_lab_core`.
    """
    dataset = make_synthetic_classification_dataset()

    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))

    print("k-NN Vote Map")
    print("Synthetic classification dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Class 0 count: {history.latest('class_0_count'):.0f}")
    print(f"Class 1 count: {history.latest('class_1_count'):.0f}")
