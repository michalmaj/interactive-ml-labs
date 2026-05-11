"""Tests for the single-tree baseline."""

import numpy as np
import pytest
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from random_forest_bagging_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SingleTreeBaseline,
    SingleTreeBaselineConfig,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)

SAMPLES_PER_CLASS: int = 20
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
PERFECT_ACCURACY: float = 1.0
STUMP_MAX_DEPTH: int = 1
XOR_SOLVING_MAX_DEPTH: int = 2


def _dataset(dataset_kind: str) -> TrainTestDataset:
    """Create a deterministic train/test dataset."""
    return make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=dataset_kind,
        ),
    )


def test_single_tree_baseline_returns_snapshot() -> None:
    """Reset should fit the baseline and return a snapshot."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=STUMP_MAX_DEPTH),
    )

    snapshot = baseline.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_single_tree_baseline_solves_axis_aligned_with_depth_one() -> None:
    """A depth-1 tree should solve low-noise axis-aligned data."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=STUMP_MAX_DEPTH),
    )

    snapshot = baseline.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert snapshot.metrics["train_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["test_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["max_depth"] == STUMP_MAX_DEPTH


def test_single_tree_baseline_depth_one_does_not_solve_xor() -> None:
    """A depth-1 tree should not solve low-noise XOR."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=STUMP_MAX_DEPTH),
    )

    snapshot = baseline.reset(_dataset(DATASET_KIND_XOR))

    assert float(snapshot.metrics["train_accuracy"]) < PERFECT_ACCURACY
    assert float(snapshot.metrics["test_accuracy"]) < PERFECT_ACCURACY


def test_single_tree_baseline_depth_two_solves_xor() -> None:
    """A depth-2 tree should solve low-noise XOR."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    snapshot = baseline.reset(_dataset(DATASET_KIND_XOR))

    assert snapshot.metrics["train_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["test_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["max_depth"] == XOR_SOLVING_MAX_DEPTH


def test_single_tree_baseline_snapshot_contains_visual_state() -> None:
    """Snapshot should expose data useful for future visualization."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    snapshot = baseline.reset(_dataset(DATASET_KIND_XOR))

    assert "train_features" in snapshot.visual_state
    assert "train_targets" in snapshot.visual_state
    assert "test_features" in snapshot.visual_state
    assert "test_targets" in snapshot.visual_state
    assert "test_predictions" in snapshot.visual_state
    assert "tree_root" in snapshot.visual_state
    assert "tree_nodes" in snapshot.visual_state


def test_single_tree_baseline_predicts_after_reset() -> None:
    """Fitted baseline should delegate predictions to the fitted tree."""
    baseline = SingleTreeBaseline(
        SingleTreeBaselineConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )
    baseline.reset(_dataset(DATASET_KIND_XOR))

    predictions = baseline.predict(
        np.array(
            [
                [-2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
                [2.0, 2.0],
            ],
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1, 0]))


def test_single_tree_baseline_rejects_usage_before_reset() -> None:
    """Using baseline before fitting should fail clearly."""
    baseline = SingleTreeBaseline()

    with pytest.raises(RuntimeError, match="reset"):
        baseline.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SingleTreeBaselineConfig(max_depth=0),
            "max_depth",
        ),
        (
            SingleTreeBaselineConfig(criterion="unknown"),
            "criterion",
        ),
    ],
)
def test_single_tree_baseline_rejects_invalid_config(
    config: SingleTreeBaselineConfig,
    expected_message: str,
) -> None:
    """Invalid baseline configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        SingleTreeBaseline(config)


def test_single_tree_baseline_rejects_test_split_without_targets() -> None:
    """Baseline evaluation requires test targets."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    dataset = TrainTestDataset(
        train=source.train,
        test=DatasetSnapshot(
            features=source.test.features,
            targets=None,
            feature_names=source.test.feature_names,
            target_names=source.test.target_names,
        ),
        metadata=source.metadata,
    )
    baseline = SingleTreeBaseline()

    with pytest.raises(ValueError, match="Test targets are required"):
        baseline.reset(dataset)


def test_single_tree_baseline_rejects_non_integer_test_targets() -> None:
    """Baseline evaluation requires integer test targets."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    dataset = TrainTestDataset(
        train=source.train,
        test=DatasetSnapshot(
            features=source.test.features,
            targets=np.asarray(source.test.targets, dtype=float),
            feature_names=source.test.feature_names,
            target_names=source.test.target_names,
        ),
        metadata=source.metadata,
    )
    baseline = SingleTreeBaseline()

    with pytest.raises(ValueError, match="Test targets must contain integers"):
        baseline.reset(dataset)
