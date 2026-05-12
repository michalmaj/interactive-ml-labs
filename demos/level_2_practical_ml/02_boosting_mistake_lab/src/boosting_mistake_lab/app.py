"""Command-line entry point for the Boosting Mistake Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import AlgorithmSnapshot, MetricsHistory

from boosting_mistake_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
)
from boosting_mistake_lab.weak_learner import WeakLearnerBaseline

CLASS_COUNT: int = 2


def main() -> None:
    """Run a minimal command-line version of the demo.

    The full boosting implementation will be added in later pull requests.
    This entry point verifies that the package can generate weighted train/test
    datasets and train a simple weak learner baseline.
    """
    axis_dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    print("Boosting Mistake Lab")
    print("Weighted datasets and weak learner baselines generated successfully.")

    _print_dataset_report("Axis-aligned", axis_dataset)
    _print_dataset_report("XOR", xor_dataset)


def _print_dataset_report(label: str, dataset: WeightedTrainTestDataset) -> None:
    """Print a short CLI report for one weighted dataset."""
    history = _build_dataset_history(dataset)
    weak_snapshot = _fit_weak_learner(dataset)

    train_features = np.asarray(dataset.train.snapshot.features, dtype=float)
    test_features = np.asarray(dataset.test.snapshot.features, dtype=float)

    print(f"{label} train features shape: {train_features.shape}")
    print(f"{label} test features shape: {test_features.shape}")
    print(f"{label} train class 0 count: {history.latest('train_class_0_count'):.0f}")
    print(f"{label} train class 1 count: {history.latest('train_class_1_count'):.0f}")
    print(f"{label} test class 0 count: {history.latest('test_class_0_count'):.0f}")
    print(f"{label} test class 1 count: {history.latest('test_class_1_count'):.0f}")
    print(f"{label} train weight sum: {history.latest('train_weight_sum'):.2f}")
    print(f"{label} test weight sum: {history.latest('test_weight_sum'):.2f}")
    print(f"{label} max train weight: {history.latest('max_train_weight'):.4f}")
    print(f"{label} min train weight: {history.latest('min_train_weight'):.4f}")
    print(f"{label} weak learner train accuracy: {weak_snapshot.metrics['train_accuracy']:.3f}")
    print(f"{label} weak learner test accuracy: {weak_snapshot.metrics['test_accuracy']:.3f}")
    print(
        f"{label} weak learner split: "
        f"x{int(weak_snapshot.metrics['feature_index']) + 1} "
        f"<= {weak_snapshot.metrics['threshold']:.3f}",
    )


def _fit_weak_learner(dataset: WeightedTrainTestDataset) -> AlgorithmSnapshot:
    """Fit a weak learner baseline and return its snapshot."""
    weak_learner = WeakLearnerBaseline()

    return weak_learner.reset(dataset)


def _build_dataset_history(dataset: WeightedTrainTestDataset) -> MetricsHistory:
    """Build simple class-count and weight metrics for one dataset."""
    train_targets = np.asarray(dataset.train.snapshot.targets, dtype=int)
    test_targets = np.asarray(dataset.test.snapshot.targets, dtype=int)

    train_class_counts = np.bincount(train_targets, minlength=CLASS_COUNT)
    test_class_counts = np.bincount(test_targets, minlength=CLASS_COUNT)

    history = MetricsHistory()
    history.add("train_class_0_count", float(train_class_counts[0]))
    history.add("train_class_1_count", float(train_class_counts[1]))
    history.add("test_class_0_count", float(test_class_counts[0]))
    history.add("test_class_1_count", float(test_class_counts[1]))
    history.add("train_weight_sum", float(np.sum(dataset.train.sample_weights)))
    history.add("test_weight_sum", float(np.sum(dataset.test.sample_weights)))
    history.add("max_train_weight", float(np.max(dataset.train.sample_weights)))
    history.add("min_train_weight", float(np.min(dataset.train.sample_weights)))

    return history
