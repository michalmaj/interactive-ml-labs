"""Tests for synthetic decision-tree dataset generation."""

import numpy as np
import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
HALF_CLASS_DISTANCE: float = CLASS_DISTANCE / 2.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
EXPECTED_CLASS_COUNT: int = 2
EXPECTED_FEATURE_COUNT: int = 2


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_decision_tree_dataset_has_expected_shape(dataset_kind: str) -> None:
    """Generated features and targets should have expected shapes."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_decision_tree_dataset(config)

    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets)

    assert features.shape == (SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT, EXPECTED_FEATURE_COUNT)
    assert targets.shape == (SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_decision_tree_dataset_is_reproducible_for_same_seed(
    dataset_kind: str,
) -> None:
    """The same seed should generate exactly the same dataset."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED,
        dataset_kind=dataset_kind,
    )

    first = make_synthetic_decision_tree_dataset(config)
    second = make_synthetic_decision_tree_dataset(config)

    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)


@pytest.mark.parametrize(
    "dataset_kind",
    [
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    ],
)
def test_synthetic_decision_tree_dataset_contains_two_balanced_classes(
    dataset_kind: str,
) -> None:
    """Generated dataset should contain two balanced classes."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        dataset_kind=dataset_kind,
    )

    dataset = make_synthetic_decision_tree_dataset(config)

    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=EXPECTED_CLASS_COUNT)

    assert set(targets.tolist()) == {0, 1}
    assert class_counts.tolist() == [SAMPLES_PER_CLASS, SAMPLES_PER_CLASS]


def test_axis_aligned_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, axis-aligned classes should be placed on the x-axis."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
    )

    dataset = make_synthetic_decision_tree_dataset(config)
    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets, dtype=int)

    class_zero_features = features[targets == 0]
    class_one_features = features[targets == 1]

    np.testing.assert_allclose(class_zero_features[:, 0], -HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_zero_features[:, 1], 0.0)
    np.testing.assert_allclose(class_one_features[:, 0], HALF_CLASS_DISTANCE)
    np.testing.assert_allclose(class_one_features[:, 1], 0.0)


def test_xor_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, XOR data should occupy diagonal class blobs."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_decision_tree_dataset(config)
    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets, dtype=int)

    class_zero_centers = set(map(tuple, np.unique(features[targets == 0], axis=0).tolist()))
    class_one_centers = set(map(tuple, np.unique(features[targets == 1], axis=0).tolist()))

    assert class_zero_centers == {
        (-HALF_CLASS_DISTANCE, -HALF_CLASS_DISTANCE),
        (HALF_CLASS_DISTANCE, HALF_CLASS_DISTANCE),
    }
    assert class_one_centers == {
        (-HALF_CLASS_DISTANCE, HALF_CLASS_DISTANCE),
        (HALF_CLASS_DISTANCE, -HALF_CLASS_DISTANCE),
    }


def test_synthetic_decision_tree_dataset_stores_metadata() -> None:
    """Generated dataset should expose useful metadata for demos."""
    config = SyntheticDecisionTreeDatasetConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
        dataset_kind=DATASET_KIND_XOR,
    )

    dataset = make_synthetic_decision_tree_dataset(config)

    assert dataset.feature_names == ("x1", "x2")
    assert dataset.target_names == ("class_0", "class_1")
    assert dataset.metadata["dataset_type"] == "synthetic_decision_tree_classification"
    assert dataset.metadata["dataset_kind"] == DATASET_KIND_XOR
    assert dataset.metadata["samples_per_class"] == SAMPLES_PER_CLASS
    assert dataset.metadata["sample_count"] == SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT
    assert dataset.metadata["class_distance"] == CLASS_DISTANCE
    assert dataset.metadata["noise_std"] == NOISE_STD_ZERO
    assert dataset.metadata["seed"] == SEED


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SyntheticDecisionTreeDatasetConfig(samples_per_class=0),
            "samples_per_class must be greater than 0",
        ),
        (
            SyntheticDecisionTreeDatasetConfig(class_distance=0.0),
            "class_distance must be greater than 0",
        ),
        (
            SyntheticDecisionTreeDatasetConfig(noise_std=-1.0),
            "noise_std cannot be negative",
        ),
        (
            SyntheticDecisionTreeDatasetConfig(dataset_kind="unknown"),
            "dataset_kind must be one of",
        ),
    ],
)
def test_synthetic_decision_tree_dataset_rejects_invalid_config(
    config: SyntheticDecisionTreeDatasetConfig,
    expected_message: str,
) -> None:
    """Invalid dataset configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_synthetic_decision_tree_dataset(config)
