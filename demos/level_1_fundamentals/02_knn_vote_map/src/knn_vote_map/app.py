"""Command-line entry point for the k-NN Vote Map demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from knn_vote_map.dataset import make_synthetic_classification_dataset
from knn_vote_map.metrics import euclidean_distances


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point currently verifies that the package can generate a dataset,
    compute distances, and use shared utilities from `ml_lab_core`.
    """
    dataset = make_synthetic_classification_dataset()

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    query_point = np.array([0.0, 0.0])
    distances = euclidean_distances(features, query_point)
    nearest_index = int(np.argmin(distances))

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))
    history.add("nearest_distance", float(distances[nearest_index]))

    print("k-NN Vote Map")
    print("Synthetic classification dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Class 0 count: {history.latest('class_0_count'):.0f}")
    print(f"Class 1 count: {history.latest('class_1_count'):.0f}")
    print(f"Query point: {query_point.tolist()}")
    print(f"Nearest point index: {nearest_index}")
    print(f"Nearest point class: {int(targets[nearest_index])}")
    print(f"Nearest distance: {history.latest('nearest_distance'):.4f}")
