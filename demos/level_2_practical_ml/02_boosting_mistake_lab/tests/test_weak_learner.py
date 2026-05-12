"""Tests for the weak learner baseline."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeakLearnerBaseline,
    WeakLearnerConfig,
    WeakLearnerSplit,
    WeightedDatasetSplit,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
    make_uniform_sample_weights,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

SAMPLES_PER_CLASS: int = 20
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
PERFECT_ACCURACY: float = 1.0


def _dataset(dataset_kind: str) -> WeightedTrainTestDataset:
    """Create a deterministic weighted train/test dataset."""
    return make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=dataset_kind,
        ),
    )


def test_weak_learner_returns_snapshot() -> None:
    """Reset should fit the weak learner and return a snapshot."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_weak_learner_exposes_fitted_split() -> None:
    """Fitted learner should expose the selected decision-stump split."""
    learner = WeakLearnerBaseline()
    learner.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert isinstance(learner.split, WeakLearnerSplit)
    assert learner.split.feature_index == 0
    assert learner.split.threshold == pytest.approx(0.0)


def test_weak_learner_solves_axis_aligned_without_noise() -> None:
    """A decision stump should solve low-noise axis-aligned data."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert snapshot.metrics["train_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["test_accuracy"] == pytest.approx(PERFECT_ACCURACY)


def test_weak_learner_does_not_solve_xor_without_noise() -> None:
    """A single decision stump should not solve XOR."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_dataset(DATASET_KIND_XOR))

    assert float(snapshot.metrics["train_accuracy"]) < PERFECT_ACCURACY
    assert float(snapshot.metrics["test_accuracy"]) < PERFECT_ACCURACY


def test_weak_learner_snapshot_contains_visual_state() -> None:
    """Snapshot should expose data needed for future boosting visualization."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_dataset(DATASET_KIND_XOR))

    assert "train_features" in snapshot.visual_state
    assert "train_targets" in snapshot.visual_state
    assert "train_predictions" in snapshot.visual_state
    assert "train_mistakes" in snapshot.visual_state
    assert "train_sample_weights" in snapshot.visual_state
    assert "test_features" in snapshot.visual_state
    assert "test_targets" in snapshot.visual_state
    assert "test_predictions" in snapshot.visual_state
    assert "test_mistakes" in snapshot.visual_state
    assert "split" in snapshot.visual_state


def test_weak_learner_predicts_after_reset() -> None:
    """Fitted weak learner should predict labels."""
    learner = WeakLearnerBaseline()
    learner.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    predictions = learner.predict(
        np.array(
            [
                [-3.0, 0.0],
                [3.0, 0.0],
            ],
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_weak_learner_accepts_single_feature_vector_for_prediction() -> None:
    """Prediction should accept one feature vector."""
    learner = WeakLearnerBaseline()
    learner.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    predictions = learner.predict(np.array([-3.0, 0.0]))

    np.testing.assert_array_equal(predictions, np.array([0]))


def test_weak_learner_rejects_usage_before_reset() -> None:
    """Using weak learner before fitting should fail clearly."""
    learner = WeakLearnerBaseline()

    with pytest.raises(RuntimeError, match="reset"):
        learner.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            WeakLearnerConfig(min_samples_leaf=0),
            "min_samples_leaf",
        ),
    ],
)
def test_weak_learner_rejects_invalid_config(
    config: WeakLearnerConfig,
    expected_message: str,
) -> None:
    """Invalid weak learner configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        WeakLearnerBaseline(config)


def test_weak_learner_rejects_train_split_without_targets() -> None:
    """Weak learner requires training targets."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    dataset = WeightedTrainTestDataset(
        train=WeightedDatasetSplit(
            snapshot=DatasetSnapshot(
                features=source.train.snapshot.features,
                targets=None,
                feature_names=source.train.snapshot.feature_names,
                target_names=source.train.snapshot.target_names,
            ),
            sample_weights=source.train.sample_weights,
        ),
        test=source.test,
        metadata=source.metadata,
    )
    learner = WeakLearnerBaseline()

    with pytest.raises(ValueError, match="train targets are required"):
        learner.reset(dataset)


def test_weak_learner_rejects_non_integer_test_targets() -> None:
    """Weak learner requires integer test targets."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    dataset = WeightedTrainTestDataset(
        train=source.train,
        test=WeightedDatasetSplit(
            snapshot=DatasetSnapshot(
                features=source.test.snapshot.features,
                targets=np.asarray(source.test.snapshot.targets, dtype=float),
                feature_names=source.test.snapshot.feature_names,
                target_names=source.test.snapshot.target_names,
            ),
            sample_weights=source.test.sample_weights,
        ),
        metadata=source.metadata,
    )
    learner = WeakLearnerBaseline()

    with pytest.raises(ValueError, match="test targets must contain integers"):
        learner.reset(dataset)


def test_weak_learner_rejects_invalid_sample_weights_shape() -> None:
    """Weak learner should validate train sample weights."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    dataset = WeightedTrainTestDataset(
        train=WeightedDatasetSplit(
            snapshot=source.train.snapshot,
            sample_weights=np.array([1.0]),
        ),
        test=source.test,
        metadata=source.metadata,
    )
    learner = WeakLearnerBaseline()

    with pytest.raises(ValueError, match="sample weights must match"):
        learner.reset(dataset)


def test_weak_learner_rejects_sample_weights_that_do_not_sum_to_one() -> None:
    """Weak learner should require normalized sample weights."""
    source = _dataset(DATASET_KIND_AXIS_ALIGNED)
    sample_count = np.asarray(source.train.snapshot.targets).shape[0]
    dataset = WeightedTrainTestDataset(
        train=WeightedDatasetSplit(
            snapshot=source.train.snapshot,
            sample_weights=np.ones(sample_count, dtype=float),
        ),
        test=source.test,
        metadata=source.metadata,
    )
    learner = WeakLearnerBaseline()

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        learner.reset(dataset)


def test_weak_learner_rejects_dataset_without_valid_split() -> None:
    """Weak learner should fail clearly when no split is possible."""
    features = np.zeros((4, 2), dtype=float)
    targets = np.array([0, 0, 1, 1], dtype=int)
    weights = make_uniform_sample_weights(4)
    split = WeightedDatasetSplit(
        snapshot=DatasetSnapshot(
            features=features,
            targets=targets,
            feature_names=("x1", "x2"),
            target_names=("class_0", "class_1"),
        ),
        sample_weights=weights,
    )
    dataset = WeightedTrainTestDataset(
        train=split,
        test=split,
        metadata={},
    )
    learner = WeakLearnerBaseline()

    with pytest.raises(ValueError, match="No valid weak learner split"):
        learner.reset(dataset)
