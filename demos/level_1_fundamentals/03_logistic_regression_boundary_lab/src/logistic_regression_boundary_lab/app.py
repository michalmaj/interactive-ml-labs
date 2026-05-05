"""Command-line entry point for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from logistic_regression_boundary_lab.dataset import (
    make_synthetic_binary_classification_dataset,
)
from logistic_regression_boundary_lab.metrics import (
    binary_cross_entropy,
    classification_metrics,
    predict_labels_from_probabilities,
    sigmoid,
)


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive implementation will be added in later pull requests.
    This entry point currently verifies that the package can generate a dataset,
    compute probabilities, threshold predictions, and report classification
    metrics.
    """
    dataset = make_synthetic_binary_classification_dataset()

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    baseline_scores = features[:, 0]
    baseline_probabilities = sigmoid(baseline_scores)
    baseline_predictions = predict_labels_from_probabilities(
        baseline_probabilities,
        threshold=0.5,
    )
    baseline_loss = binary_cross_entropy(targets, baseline_probabilities)
    baseline_metrics = classification_metrics(targets, baseline_predictions)

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))
    history.add("baseline_loss", baseline_loss)
    history.add("baseline_accuracy", baseline_metrics.accuracy)
    history.add("baseline_precision", baseline_metrics.precision)
    history.add("baseline_recall", baseline_metrics.recall)

    print("Logistic Regression Boundary Lab")
    print("Synthetic binary classification dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Class 0 count: {history.latest('class_0_count'):.0f}")
    print(f"Class 1 count: {history.latest('class_1_count'):.0f}")
    print("Baseline score: x1")
    print("Baseline threshold: 0.50")
    print(f"Baseline loss: {history.latest('baseline_loss'):.4f}")
    print(f"Baseline accuracy: {history.latest('baseline_accuracy'):.4f}")
    print(f"Baseline precision: {history.latest('baseline_precision'):.4f}")
    print(f"Baseline recall: {history.latest('baseline_recall'):.4f}")
