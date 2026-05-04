"""Command-line entry point for the k-NN Vote Map demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from knn_vote_map.classifier import KNearestNeighborsClassifier, KNearestNeighborsConfig
from knn_vote_map.dataset import make_synthetic_classification_dataset


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive implementation will be added in later pull requests.
    This entry point currently verifies that the package can generate a dataset,
    fit a k-NN classifier, classify one query point, and use shared utilities
    from `ml_lab_core`.
    """
    dataset = make_synthetic_classification_dataset()

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=5))
    classifier.fit(dataset)

    query_point = np.array([0.0, 0.0])
    prediction = classifier.predict_one(query_point)

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))
    history.add("predicted_label", float(prediction.predicted_label))

    print("k-NN Vote Map")
    print("Synthetic classification dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Class 0 count: {history.latest('class_0_count'):.0f}")
    print(f"Class 1 count: {history.latest('class_1_count'):.0f}")
    print(f"Query point: {query_point.tolist()}")
    print(f"k: {classifier.k}")
    print(f"Predicted class: {prediction.predicted_label}")
    print(f"Vote counts: {prediction.vote_counts}")
    print("Nearest neighbors:")

    for neighbor in prediction.neighbors:
        print(
            f"  index={neighbor.index}, class={neighbor.label}, distance={neighbor.distance:.4f}",
        )
