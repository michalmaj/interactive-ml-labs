"""Tests for synthetic train/test dataset generation."""

import numpy as np
import pytest
from random_forest_bagging_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    make_synthetic_train_test_dataset,
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
def test_synthetic_train_test_dataset_has_expected_shapes(dataset_kind: str) -> None:
    """Generated train and test splits should have expected shapes."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_train_test_dataset(config)

    train_features = np.asarray(dataset.train.features)
    train_targets = np.asarray(dataset.train.targets)
    test_features = np.asarray(dataset.test.features)
    test_targets = np.asarray(dataset.test.targets)

    assert train_features.shape == (
        TRAIN_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,
        EXPECTED_FEATURE_COUNT,
    )
    assert train_targets.shape == (TRAIN_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)
    assert test_features.shape == (
        TEST_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,
        EXPECTED_FEATURE_COUNT,
    )
    assert test_targets.shape == (TEST_SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_train_test_dataset_is_reproducible(dataset_kind: str) -> None:
    """The same seed should generate exactly the same train/test dataset."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        seed=SEED,
        dataset_kind=dataset_kind,
    )

    first = make_synthetic_train_test_dataset(config)
    second = make_synthetic_train_test_dataset(config)

    np.testing.assert_allclose(first.train.features, second.train.features)
    np.testing.assert_array_equal(first.train.targets, second.train.targets)
    np.testing.assert_allclose(first.test.features, second.test.features)
    np.testing.assert_array_equal(first.test.targets, second.test.targets)


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_train_test_dataset_contains_balanced_classes(dataset_kind: str) -> None:
    """Both splits should contain balanced binary classes."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_train_test_dataset(config)

    train_targets = np.asarray(dataset.train.targets, dtype=int)
    test_targets = np.asarray(dataset.test.targets, dtype=int)

    train_class_counts = np.bincount(train_targets, minlength=EXPECTED_CLASS_COUNT)
    test_class_counts = np.bincount(test_targets, minlength=EXPECTED_CLASS_COUNT)

    assert set(train_targets.tolist()) == {CLASS_ZERO_LABEL, CLASS_ONE_LABEL}
    assert set(test_targets.tolist()) == {CLASS_ZERO_LABEL, CLASS_ONE_LABEL}
    assert train_class_counts.tolist() == [TRAIN_SAMPLES_PER_CLASS, TRAIN_SAMPLES_PER_CLASS]
    assert test_class_counts.tolist() == [TEST_SAMPLES_PER_CLASS, TEST_SAMPLES_PER_CLASS]


def test_axis_aligned_train_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, axis-aligned train classes should be placed on the x-axis."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
    )

    dataset = make_synthetic_train_test_dataset(config)
    features = np.asarray(dataset.train.features)
    targets = np.asarray(dataset.train.targets, dtype=int)

    class_zero_features = features[targets == CLASS_ZERO_LABEL]
    class_one_features = features[targets == CLASS_ONE_LABEL]

    np.testing.assert_allclose(class_zero_features[:, 0], -HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_zero_features[:, 1], 0.0)
    np.testing.assert_allclose(class_one_features[:, 0], HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_one_features[:, 1], 0.0)


def test_xor_train_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, XOR train data should occupy diagonal class blobs."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_train_test_dataset(config)
    features = np.asarray(dataset.train.features)
    targets = np.asarray(dataset.train.targets, dtype=int)

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


def test_synthetic_train_test_dataset_stores_metadata() -> None:
    """Generated train/test dataset should expose useful metadata."""
    config = SyntheticTrainTestDatasetConfig(
        train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
        test_samples_per_class=TEST_SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_train_test_dataset(config)

    assert dataset.train.feature_names == ("x1", "x2")
    assert dataset.test.feature_names == ("x1", "x2")
    assert dataset.train.target_names == ("class_0", "class_1")
    assert dataset.test.target_names == ("class_0", "class_1")
    assert dataset.metadata["dataset_type"] == "synthetic_random_forest_classification"
    assert dataset.metadata["dataset_kind"] == DATASET_KIND_XOR
    assert dataset.metadata["train_samples_per_class"] == TRAIN_SAMPLES_PER_CLASS
    assert dataset.metadata["test_samples_per_class"] == TEST_SAMPLES_PER_CLASS
    assert dataset.metadata["class_distance"] == CLASS_DISTANCE
    assert dataset.metadata["noise_std"] == NOISE_STD_ZERO
    assert dataset.metadata["seed"] == SEED
    assert dataset.train.metadata["split"] == "train"
    assert dataset.test.metadata["split"] == "test"


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SyntheticTrainTestDatasetConfig(train_samples_per_class=0),
            "train_samples_per_class must be greater than 0",
        ),
        (
            SyntheticTrainTestDatasetConfig(test_samples_per_class=0),
            "test_samples_per_class must be greater than 0",
        ),
        (
            SyntheticTrainTestDatasetConfig(class_distance=0.0),
            "class_distance must be greater than 0",
        ),
        (
            SyntheticTrainTestDatasetConfig(noise_std=-1.0),
            "noise_std cannot be negative",
        ),
        (
            SyntheticTrainTestDatasetConfig(dataset_kind="unknown"),
            "dataset_kind must be one of",
        ),
    ],
)
def test_synthetic_train_test_dataset_rejects_invalid_config(
    config: SyntheticTrainTestDatasetConfig,
    expected_message: str,
) -> None:
    """Invalid dataset configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_synthetic_train_test_dataset(config)
