"""Tests for synthetic weighted dataset generation."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    make_synthetic_weighted_dataset,
    make_uniform_sample_weights,
)

TRAIN_SAMPLES_PER_CLASS: int = 12
TEST_SAMPLES_PER_CLASS: int = 8
CLASS_DISTANCE: float = 6.0
HALF_CLASS_DISTANCE: float = CLASS_DISTANCE / 2.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
EXPECTED_CLASS_COUNT: int = 2
EXPECTED_FEATURE_COUNT: int = 2
CLASS_ZERO_LABEL: int = 0
CLASS_ONE_LABEL: int = 1


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_weighted_dataset_has_expected_shapes(dataset_kind: str) -> None:
    """Generated train and test splits should have expected shapes."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_weighted_dataset(config)

    train_features = np.asarray(dataset.train.snapshot.features)
    train_targets = np.asarray(dataset.train.snapshot.targets)
    test_features = np.asarray(dataset.test.snapshot.features)
    test_targets = np.asarray(dataset.test.snapshot.targets)

    assert train_features.shape == (
        TRAIN_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,
        EXPECTED_FEATURE_COUNT,
    )
    assert train_targets.shape == (TRAIN_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)
    assert dataset.train.sample_weights.shape == train_targets.shape

    assert test_features.shape == (
        TEST_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,
        EXPECTED_FEATURE_COUNT,
    )
    assert test_targets.shape == (TEST_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)
    assert dataset.test.sample_weights.shape == test_targets.shape


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_weighted_dataset_is_reproducible(dataset_kind: str) -> None:
    """The same seed should generate exactly the same weighted dataset."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        seed=SEED,
        dataset_kind=dataset_kind,
    )

    first = make_synthetic_weighted_dataset(config)
    second = make_synthetic_weighted_dataset(config)

    np.testing.assert_allclose(first.train.snapshot.features, second.train.snapshot.features)
    np.testing.assert_array_equal(first.train.snapshot.targets, second.train.snapshot.targets)
    np.testing.assert_allclose(first.train.sample_weights, second.train.sample_weights)
    np.testing.assert_allclose(first.test.snapshot.features, second.test.snapshot.features)
    np.testing.assert_array_equal(first.test.snapshot.targets, second.test.snapshot.targets)
    np.testing.assert_allclose(first.test.sample_weights, second.test.sample_weights)


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_weighted_dataset_contains_balanced_classes(dataset_kind: str) -> None:
    """Both splits should contain balanced binary classes."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_weighted_dataset(config)

    train_targets = np.asarray(dataset.train.snapshot.targets, dtype=int)
    test_targets = np.asarray(dataset.test.snapshot.targets, dtype=int)

    train_class_counts = np.bincount(train_targets, minlength=EXPECTED_CLASS_COUNT)
    test_class_counts = np.bincount(test_targets, minlength=EXPECTED_CLASS_COUNT)

    assert set(train_targets.tolist()) == {CLASS_ZERO_LABEL, CLASS_ONE_LABEL}
    assert set(test_targets.tolist()) == {CLASS_ZERO_LABEL, CLASS_ONE_LABEL}
    assert train_class_counts.tolist() == [TRAIN_SAMPLES_PER_CLASS, TRAIN_SAMPLES_PER_CLASS]
    assert test_class_counts.tolist() == [TEST_SAMPLES_PER_CLASS, TEST_SAMPLES_PER_CLASS]


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_weighted_dataset_has_normalized_weights(dataset_kind: str) -> None:
    """Initial train and test weights should be uniform and normalized."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_weighted_dataset(config)

    assert np.sum(dataset.train.sample_weights) == pytest.approx(1.0)
    assert np.sum(dataset.test.sample_weights) == pytest.approx(1.0)
    assert np.unique(dataset.train.sample_weights).shape == (1,)
    assert np.unique(dataset.test.sample_weights).shape == (1,)


def test_axis_aligned_train_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, axis-aligned train classes should be placed on the x-axis."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
    )

    dataset = make_synthetic_weighted_dataset(config)
    features = np.asarray(dataset.train.snapshot.features)
    targets = np.asarray(dataset.train.snapshot.targets, dtype=int)

    class_zero_features = features[targets == CLASS_ZERO_LABEL]
    class_one_features = features[targets == CLASS_ONE_LABEL]

    np.testing.assert_allclose(class_zero_features[:, 0], -HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_zero_features[:, 1], 0.0)
    np.testing.assert_allclose(class_one_features[:, 0], HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_one_features[:, 1], 0.0)


def test_xor_train_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, XOR train data should occupy diagonal class blobs."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_weighted_dataset(config)
    features = np.asarray(dataset.train.snapshot.features)
    targets = np.asarray(dataset.train.snapshot.targets, dtype=int)

    class_zero_centers = set(
        map(tuple, np.unique(features[targets == CLASS_ZERO_LABEL], axis=0).tolist()),
    )
    class_one_centers = set(
        map(tuple, np.unique(features[targets == CLASS_ONE_LABEL], axis=0).tolist()),
    )

    assert class_zero_centers == {
        (-HALF_CLASS_DISTANCE, -HALF_CLASS_DISTANCE),
        (HALF_CLASS_DISTANCE, HALF_CLASS_DISTANCE),
    }
    assert class_one_centers == {
        (-HALF_CLASS_DISTANCE, HALF_CLASS_DISTANCE),
        (HALF_CLASS_DISTANCE, -HALF_CLASS_DISTANCE),
    }


def test_synthetic_weighted_dataset_stores_metadata() -> None:
    """Generated weighted dataset should expose useful metadata."""
    config = SyntheticWeightedDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_weighted_dataset(config)

    assert dataset.train.snapshot.feature_names == ("x1", "x2")
    assert dataset.test.snapshot.feature_names == ("x1", "x2")
    assert dataset.train.snapshot.target_names == ("class_0", "class_1")
    assert dataset.test.snapshot.target_names == ("class_0", "class_1")
    assert dataset.metadata["dataset_type"] == "synthetic_boosting_classification"
    assert dataset.metadata["dataset_kind"] == DATASET_KIND_XOR
    assert dataset.metadata["train_samples_per_class"] == TRAIN_SAMPLES_PER_CLASS
    assert dataset.metadata["test_samples_per_class"] == TEST_SAMPLES_PER_CLASS
    assert dataset.metadata["class_distance"] == CLASS_DISTANCE
    assert dataset.metadata["noise_std"] == NOISE_STD_ZERO
    assert dataset.metadata["seed"] == SEED
    assert dataset.train.snapshot.metadata["split"] == "train"
    assert dataset.test.snapshot.metadata["split"] == "test"
    assert dataset.train.snapshot.metadata["sample_weight_sum"] == pytest.approx(1.0)
    assert dataset.test.snapshot.metadata["sample_weight_sum"] == pytest.approx(1.0)


def test_make_uniform_sample_weights_returns_normalized_uniform_weights() -> None:
    """Uniform sample weights should sum to 1.0."""
    weights = make_uniform_sample_weights(4)

    np.testing.assert_allclose(weights, np.array([0.25, 0.25, 0.25, 0.25]))
    assert np.sum(weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "sample_count",
    [
        0,
        -1,
    ],
)
def test_make_uniform_sample_weights_rejects_invalid_sample_count(sample_count: int) -> None:
    """Uniform weights require a positive sample count."""
    with pytest.raises(ValueError, match="sample_count"):
        make_uniform_sample_weights(sample_count)


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SyntheticWeightedDatasetConfig(train_samples_per_class=0),
            "train_samples_per_class must be greater than 0",
        ),
        (
            SyntheticWeightedDatasetConfig(test_samples_per_class=0),
            "test_samples_per_class must be greater than 0",
        ),
        (
            SyntheticWeightedDatasetConfig(class_distance=0.0),
            "class_distance must be greater than 0",
        ),
        (
            SyntheticWeightedDatasetConfig(noise_std=-1.0),
            "noise_std cannot be negative",
        ),
        (
            SyntheticWeightedDatasetConfig(dataset_kind="unknown"),
            "dataset_kind must be one of",
        ),
    ],
)
def test_synthetic_weighted_dataset_rejects_invalid_config(
    config: SyntheticWeightedDatasetConfig,
    expected_message: str,
) -> None:
    """Invalid dataset configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_synthetic_weighted_dataset(config)
