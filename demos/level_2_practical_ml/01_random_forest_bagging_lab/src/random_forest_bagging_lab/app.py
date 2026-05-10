"""Command-line entry point for the Random Forest Bagging Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    make_synthetic_train_test_dataset,
)

CLASS_COUNT: int = 2


def main() -> None:
    """Run a minimal command-line version of the demo.

    The random forest implementation will be added in later pull requests.
    This entry point verifies that the package can generate train/test datasets.
    """
    axis_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    axis_history = _build_dataset_history(axis_dataset.train.targets, axis_dataset.test.targets)
    xor_history = _build_dataset_history(xor_dataset.train.targets, xor_dataset.test.targets)

    print("Random Forest Bagging Lab")
    print("Synthetic train/test datasets generated successfully.")

    _print_dataset_summary(
        label="Axis-aligned",
        train_features=np.asarray(axis_dataset.train.features, dtype=float),
        test_features=np.asarray(axis_dataset.test.features, dtype=float),
        history=axis_history,
    )
    _print_dataset_summary(
        label="XOR",
        train_features=np.asarray(xor_dataset.train.features, dtype=float),
        test_features=np.asarray(xor_dataset.test.features, dtype=float),
        history=xor_history,
    )


def _build_dataset_history(train_targets: object, test_targets: object) -> MetricsHistory:
    """Build simple class-count metrics for one train/test dataset."""
    train_class_counts = np.bincount(
        np.asarray(train_targets, dtype=int),
        minlength=CLASS_COUNT,
    )
    test_class_counts = np.bincount(
        np.asarray(test_targets, dtype=int),
        minlength=CLASS_COUNT,
    )

    history = MetricsHistory()
    history.add("train_class_0_count", float(train_class_counts[0]))
    history.add("train_class_1_count", float(train_class_counts[1]))
    history.add("test_class_0_count", float(test_class_counts[0]))
    history.add("test_class_1_count", float(test_class_counts[1]))

    return history


def _print_dataset_summary(
    *,
    label: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    history: MetricsHistory,
) -> None:
    """Print a short CLI summary for one dataset."""
    print(f"{label} train features shape: {train_features.shape}")
    print(f"{label} test features shape: {test_features.shape}")
    print(f"{label} train class 0 count: {history.latest('train_class_0_count'):.0f}")
    print(f"{label} train class 1 count: {history.latest('train_class_1_count'):.0f}")
    print(f"{label} test class 0 count: {history.latest('test_class_0_count'):.0f}")
    print(f"{label} test class 1 count: {history.latest('test_class_1_count'):.0f}")