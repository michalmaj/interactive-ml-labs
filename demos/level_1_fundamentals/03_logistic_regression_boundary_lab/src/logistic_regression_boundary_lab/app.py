"""Command-line entry point for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from logistic_regression_boundary_lab.algorithm import (
    LogisticRegressionConfig,
    StepwiseLogisticRegression,
)
from logistic_regression_boundary_lab.dataset import (
    make_synthetic_binary_classification_dataset,
)

DEMO_STEPS: int = 25


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive implementation will be added in later pull requests.
    This entry point verifies that the package can generate a dataset, train a
    logistic regression model step by step, and report classification metrics.
    """
    dataset = make_synthetic_binary_classification_dataset()

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=2)

    model = StepwiseLogisticRegression(
        LogisticRegressionConfig(
            learning_rate=0.1,
            max_steps=DEMO_STEPS,
            threshold=0.5,
        ),
    )
    model.reset(dataset)

    initial_snapshot = model.snapshot()

    for _ in range(DEMO_STEPS):
        snapshot = model.step()

        if snapshot.done:
            break

    final_snapshot = model.snapshot()

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("class_0_count", float(class_counts[0]))
    history.add("class_1_count", float(class_counts[1]))
    history.add("initial_loss", float(initial_snapshot.metrics["loss"]))
    history.add("final_loss", float(final_snapshot.metrics["loss"]))
    history.add("final_accuracy", float(final_snapshot.metrics["accuracy"]))
    history.add("final_precision", float(final_snapshot.metrics["precision"]))
    history.add("final_recall", float(final_snapshot.metrics["recall"]))

    print("Logistic Regression Boundary Lab")
    print("Synthetic binary classification dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Class 0 count: {history.latest('class_0_count'):.0f}")
    print(f"Class 1 count: {history.latest('class_1_count'):.0f}")
    print(f"Training steps: {final_snapshot.iteration}")
    print(f"Initial loss: {history.latest('initial_loss'):.4f}")
    print(f"Final loss: {history.latest('final_loss'):.4f}")
    print(f"Final accuracy: {history.latest('final_accuracy'):.4f}")
    print(f"Final precision: {history.latest('final_precision'):.4f}")
    print(f"Final recall: {history.latest('final_recall'):.4f}")
    print(f"Weight 1: {float(final_snapshot.metrics['weight_1']):.4f}")
    print(f"Weight 2: {float(final_snapshot.metrics['weight_2']):.4f}")
    print(f"Bias: {float(final_snapshot.metrics['bias']):.4f}")
