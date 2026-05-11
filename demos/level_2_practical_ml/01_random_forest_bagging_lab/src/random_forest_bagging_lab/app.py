"""Command-line entry point for the Random Forest Bagging Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import MetricsHistory

from random_forest_bagging_lab.bootstrap import (
    BootstrapSampleConfig,
    make_bootstrap_sample,
)
from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)

CLASS_COUNT: int = 2
BOOTSTRAP_SEED: int = 7


def main() -> None:
    """Run a minimal command-line version of the demo.

    The random forest implementation will be added in later pull requests.
    This entry point verifies that the package can generate train/test datasets
    and draw bootstrap samples from training data.
    """
    axis_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    print("Random Forest Bagging Lab")
    print("Synthetic train/test datasets and bootstrap samples generated successfully.")

    _print_dataset_report("Axis-aligned", axis_dataset)
    _print_dataset_report("XOR", xor_dataset)


def _print_dataset_report(label: str, dataset: TrainTestDataset) -> None:
    """Print a short CLI report for one train/test dataset."""
    history = _build_dataset_history(dataset)
    bootstrap = make_bootstrap_sample(
        dataset.train,
        BootstrapSampleConfig(seed=BOOTSTRAP_SEED),
    )

    train_features = np.asarray(dataset.train.features, dtype=float)
    test_features = np.asarray(dataset.test.features, dtype=float)

    print(f"{label} train features shape: {train_features.shape}")
    print(f"{label} test features shape: {test_features.shape}")
    print(f"{label} train class 0 count: {history.latest('train_class_0_count'):.0f}")
    print(f"{label} train class 1 count: {history.latest('train_class_1_count'):.0f}")
    print(f"{label} test class 0 count: {history.latest('test_class_0_count'):.0f}")
    print(f"{label} test class 1 count: {history.latest('test_class_1_count'):.0f}")
    print(f"{label} bootstrap draw count: {bootstrap.sample_indices.shape[0]}")
    print(f"{label} bootstrap unique samples: {bootstrap.unique_sample_count}")
    print(f"{label} bootstrap OOB samples: {bootstrap.oob_indices.shape[0]}")


def _build_dataset_history(dataset: TrainTestDataset) -> MetricsHistory:
    """Build simple class-count metrics for one train/test dataset."""
    train_class_counts = np.bincount(
        np.asarray(dataset.train.targets, dtype=int),
        minlength=CLASS_COUNT,
    )
    test_class_counts = np.bincount(
        np.asarray(dataset.test.targets, dtype=int),
        minlength=CLASS_COUNT,
    )

    history = MetricsHistory()
    history.add("train_class_0_count", float(train_class_counts[0]))
    history.add("train_class_1_count", float(train_class_counts[1]))
    history.add("test_class_0_count", float(test_class_counts[0]))
    history.add("test_class_1_count", float(test_class_counts[1]))

    return history
