"""Command-line entry point for the Gradient Descent Playground demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from gradient_descent_playground.dataset import make_synthetic_regression_dataset
from gradient_descent_playground.metrics import mean_squared_error


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point currently verifies that the package can generate a dataset,
    compute a regression metric, and use shared utilities from `ml_lab_core`.
    """
    dataset = make_synthetic_regression_dataset()

    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets)

    baseline_predictions = np.zeros_like(targets)
    baseline_mse = mean_squared_error(targets, baseline_predictions)

    history = MetricsHistory()
    history.add("sample_count", float(features.shape[0]))
    history.add("baseline_mse", baseline_mse)

    print("Gradient Descent Playground")
    print("Synthetic dataset generated successfully.")
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample count metric: {history.latest('sample_count'):.0f}")
    print(f"Baseline MSE: {history.latest('baseline_mse'):.4f}")
