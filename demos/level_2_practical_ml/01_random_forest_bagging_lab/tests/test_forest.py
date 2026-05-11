"""Tests for the random forest model."""

import numpy as np
import pytest
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from random_forest_bagging_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    ForestTreeMember,
    RandomForestConfig,
    RandomForestModel,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)

SAMPLES_PER_CLASS: int = 40
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
TREE_COUNT: int = 11
STUMP_MAX_DEPTH: int = 1
XOR_SOLVING_MAX_DEPTH: int = 2
PERFECT_ACCURACY: float = 1.0
HIGH_ACCURACY: float = 0.95


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


def test_random_forest_returns_snapshot() -> None:
    """Reset should fit the forest and return a snapshot."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=STUMP_MAX_DEPTH,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_random_forest_fits_expected_number_of_trees() -> None:
    """Forest should contain the configured number of trees."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=STUMP_MAX_DEPTH,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert len(forest.members) == TREE_COUNT
    assert snapshot.metrics["tree_count"] == TREE_COUNT
    assert all(isinstance(member, ForestTreeMember) for member in forest.members)


def test_random_forest_solves_axis_aligned_with_depth_one() -> None:
    """A forest of shallow trees should solve low-noise axis-aligned data."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=STUMP_MAX_DEPTH,
            seed=SEED,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert snapshot.metrics["train_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["test_accuracy"] == pytest.approx(PERFECT_ACCURACY)


def test_random_forest_depth_one_does_not_solve_xor() -> None:
    """A depth-1 forest should still be too weak for XOR."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=STUMP_MAX_DEPTH,
            seed=SEED,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_XOR))

    assert float(snapshot.metrics["train_accuracy"]) < PERFECT_ACCURACY
    assert float(snapshot.metrics["test_accuracy"]) < PERFECT_ACCURACY


def test_random_forest_solves_xor_with_depth_two() -> None:
    """A depth-2 forest should solve low-noise XOR with high accuracy."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=XOR_SOLVING_MAX_DEPTH,
            seed=SEED,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_XOR))

    assert float(snapshot.metrics["train_accuracy"]) >= HIGH_ACCURACY
    assert float(snapshot.metrics["test_accuracy"]) >= HIGH_ACCURACY


def test_random_forest_snapshot_contains_visual_state() -> None:
    """Snapshot should expose data useful for future visualization."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=XOR_SOLVING_MAX_DEPTH,
        ),
    )

    snapshot = forest.reset(_dataset(DATASET_KIND_XOR))

    assert "train_features" in snapshot.visual_state
    assert "train_targets" in snapshot.visual_state
    assert "test_features" in snapshot.visual_state
    assert "test_targets" in snapshot.visual_state
    assert "train_predictions" in snapshot.visual_state
    assert "test_predictions" in snapshot.visual_state
    assert "train_confidence" in snapshot.visual_state
    assert "test_confidence" in snapshot.visual_state
    assert "train_vote_counts" in snapshot.visual_state
    assert "test_vote_counts" in snapshot.visual_state
    assert "members" in snapshot.visual_state


def test_random_forest_predicts_labels() -> None:
    """Fitted forest should predict labels through majority voting."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=XOR_SOLVING_MAX_DEPTH,
            seed=SEED,
        ),
    )
    forest.reset(_dataset(DATASET_KIND_XOR))

    predictions = forest.predict(
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


def test_random_forest_predicts_with_confidence() -> None:
    """Forest should expose voting confidence for predictions."""
    forest = RandomForestModel(
        RandomForestConfig(
            tree_count=TREE_COUNT,
            max_depth=XOR_SOLVING_MAX_DEPTH,
            seed=SEED,
        ),
    )
    forest.reset(_dataset(DATASET_KIND_XOR))

    result = forest.predict_with_confidence(
        np.array(
            [
                [-2.0, -2.0],
                [2.0, 2.0],
            ],
        ),
    )

    assert result.predictions.shape == (2,)
    assert result.confidence.shape == (2,)
    assert result.vote_counts.shape == (2, 2)
    assert np.all(result.confidence >= 0.0)
    assert np.all(result.confidence <= 1.0)


def test_random_forest_rejects_usage_before_reset() -> None:
    """Using forest before fitting should fail clearly."""
    forest = RandomForestModel()

    with pytest.raises(RuntimeError, match="reset"):
        forest.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            RandomForestConfig(tree_count=0),
            "tree_count",
        ),
        (
            RandomForestConfig(max_depth=0),
            "max_depth",
        ),
        (
            RandomForestConfig(criterion="unknown"),
            "criterion",
        ),
        (
            RandomForestConfig(bootstrap_sample_ratio=0.0),
            "bootstrap_sample_ratio",
        ),
        (
            RandomForestConfig(bootstrap_sample_ratio=1.1),
            "bootstrap_sample_ratio",
        ),
        (
            RandomForestConfig(min_samples_split=1),
            "min_samples_split",
        ),
        (
            RandomForestConfig(min_samples_leaf=0),
            "min_samples_leaf",
        ),
        (
            RandomForestConfig(min_information_gain=-1.0),
            "min_information_gain",
        ),
    ],
)
def test_random_forest_rejects_invalid_config(
    config: RandomForestConfig,
    expected_message: str,
) -> None:
    """Invalid forest configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        RandomForestModel(config)


def test_random_forest_rejects_test_split_without_targets() -> None:
    """Forest evaluation requires test targets."""
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
    forest = RandomForestModel()

    with pytest.raises(ValueError, match="test targets are required"):
        forest.reset(dataset)


def test_random_forest_rejects_non_integer_test_targets() -> None:
    """Forest evaluation requires integer test targets."""
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
    forest = RandomForestModel()

    with pytest.raises(ValueError, match="test targets must contain integers"):
        forest.reset(dataset)
