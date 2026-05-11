"""Command-line entry point for the Random Forest Bagging Lab demo."""

from __future__ import annotations

import numpy as np
from ml_lab_core import AlgorithmSnapshot, MetricsHistory

from random_forest_bagging_lab.baseline import (
    SingleTreeBaseline,
    SingleTreeBaselineConfig,
)
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
from random_forest_bagging_lab.forest import RandomForestConfig, RandomForestModel
from random_forest_bagging_lab.report import (
    build_model_comparison_report,
    format_model_comparison_report,
)
from random_forest_bagging_lab.voting import majority_vote

CLASS_COUNT: int = 2
BOOTSTRAP_SEED: int = 7
BASELINE_MAX_DEPTH: int = 2
FOREST_TREE_COUNT: int = 25
FOREST_MAX_DEPTH: int = 2


def main() -> None:
    """Run a command-line comparison report for the demo.

    This entry point verifies that the package can generate train/test datasets,
    draw bootstrap samples, combine example predictions through voting, evaluate
    a single-tree baseline, fit a random forest model, and print a readable
    comparison report.
    """
    axis_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    voting_result = majority_vote(
        np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [1, 1, 1, 0],
            ],
            dtype=int,
        ),
        class_count=CLASS_COUNT,
    )

    print("Random Forest Bagging Lab")
    print(
        "Synthetic train/test datasets, bootstrap samples, voting, baseline tree, "
        "random forest, and comparison report generated successfully.",
    )

    _print_dataset_report("Axis-aligned", axis_dataset)
    _print_dataset_report("XOR", xor_dataset)
    _print_voting_report(voting_result.predictions, voting_result.confidence)


def _print_dataset_report(label: str, dataset: TrainTestDataset) -> None:
    """Print a readable CLI report for one train/test dataset."""
    history = _build_dataset_history(dataset)
    bootstrap = make_bootstrap_sample(
        dataset.train,
        BootstrapSampleConfig(seed=BOOTSTRAP_SEED),
    )
    baseline_snapshot = _fit_single_tree_baseline(dataset)
    forest_snapshot = _fit_random_forest(dataset)
    comparison_report = build_model_comparison_report(
        single_tree_snapshot=baseline_snapshot,
        forest_snapshot=forest_snapshot,
    )

    train_features = np.asarray(dataset.train.features, dtype=float)
    test_features = np.asarray(dataset.test.features, dtype=float)

    print("")
    print(f"{label} dataset:")
    print(f"  train features shape: {train_features.shape}")
    print(f"  test features shape: {test_features.shape}")
    print(f"  train class 0 count: {history.latest('train_class_0_count'):.0f}")
    print(f"  train class 1 count: {history.latest('train_class_1_count'):.0f}")
    print(f"  test class 0 count: {history.latest('test_class_0_count'):.0f}")
    print(f"  test class 1 count: {history.latest('test_class_1_count'):.0f}")

    print(f"{label} bootstrap:")
    print(f"  draw count: {bootstrap.sample_indices.shape[0]}")
    print(f"  unique samples: {bootstrap.unique_sample_count}")
    print(f"  OOB samples: {bootstrap.oob_indices.shape[0]}")

    for line in format_model_comparison_report(
        label=label,
        report=comparison_report,
    ):
        print(line)


def _fit_single_tree_baseline(dataset: TrainTestDataset) -> AlgorithmSnapshot:
    """Fit a single-tree baseline and return its snapshot."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=BASELINE_MAX_DEPTH),
    )

    return baseline.reset(dataset)


def _fit_random_forest(dataset: TrainTestDataset) -> AlgorithmSnapshot:
    """Fit a random forest and return its snapshot."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=FOREST_TREE_COUNT,
            max_depth=FOREST_MAX_DEPTH,
        ),
    )

    return forest.reset(dataset)


def _print_voting_report(predictions: np.ndarray, confidence: np.ndarray) -> None:
    """Print a short voting summary."""
    mean_confidence = float(np.mean(confidence))

    print("")
    print("Example majority vote:")
    print(f"  predictions: {predictions.tolist()}")
    print(f"  confidence: {confidence.round(3).tolist()}")
    print(f"  mean confidence: {mean_confidence:.3f}")


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
