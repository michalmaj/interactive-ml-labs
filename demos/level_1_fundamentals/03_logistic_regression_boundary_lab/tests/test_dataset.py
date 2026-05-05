"""Tests for synthetic binary classification dataset generation."""

import numpy as np
import pytest
from logistic_regression_boundary_lab import (
    SyntheticBinaryClassificationConfig,
    make_synthetic_binary_classification_dataset,
)

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 5.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
EXPECTED_CLASS_COUNT: int = 2
EXPECTED_FEATURE_COUNT: int = 2


def test_synthetic_binary_classification_dataset_has_expected_shape() -> None:
    """Generated features and targets should have expected shapes."""
    config = SyntheticBinaryClassificationConfig(samples_per_class=SAMPLES_PER_CLASS)
    dataset = make_synthetic_binary_classification_dataset(config)

    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets)

    assert features.shape == (SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT, EXPECTED_FEATURE_COUNT)
    assert targets.shape == (SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT,)


def test_synthetic_binary_classification_dataset_is_reproducible_for_same_seed() -> None:
    """The same seed should generate exactly the same dataset."""
    config = SyntheticBinaryClassificationConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        seed=SEED,
    )

    first = make_synthetic_binary_classification_dataset(config)
    second = make_synthetic_binary_classification_dataset(config)

    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_array_equal(first.targets, second.targets)


def test_synthetic_binary_classification_dataset_contains_two_balanced_classes() -> None:
    """Generated dataset should contain two balanced classes."""
    config = SyntheticBinaryClassificationConfig(samples_per_class=SAMPLES_PER_CLASS)
    dataset = make_synthetic_binary_classification_dataset(config)

    targets = np.asarray(dataset.targets, dtype=int)
    class_counts = np.bincount(targets, minlength=EXPECTED_CLASS_COUNT)

    assert set(targets.tolist()) == {0, 1}
    assert class_counts.tolist() == [SAMPLES_PER_CLASS, SAMPLES_PER_CLASS]


def test_synthetic_binary_classification_dataset_has_expected_centers_without_noise() -> None:
    """Without noise, each class should be generated exactly at its center."""
    config = SyntheticBinaryClassificationConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
    )

    dataset = make_synthetic_binary_classification_dataset(config)
    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets, dtype=int)

    class_zero_features = features[targets == 0]
    class_one_features = features[targets == 1]

    np.testing.assert_allclose(class_zero_features[:, 0], -CLASS_DISTANCE / 2.0)
    np.testing.assert_allclose(class_zero_features[:, 1], 0.0)
    np.testing.assert_allclose(class_one_features[:, 0], CLASS_DISTANCE / 2.0)
    np.testing.assert_allclose(class_one_features[:, 1], 0.0)


def test_synthetic_binary_classification_dataset_stores_metadata() -> None:
    """Generated dataset should expose useful metadata for demos."""
    config = SyntheticBinaryClassificationConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        class_distance=CLASS_DISTANCE,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
    )

    dataset = make_synthetic_binary_classification_dataset(config)

    assert dataset.feature_names == ("x1", "x2")
    assert dataset.target_names == ("class_0", "class_1")
    assert dataset.metadata["dataset_type"] == "synthetic_binary_classification"
    assert dataset.metadata["samples_per_class"] == SAMPLES_PER_CLASS
    assert dataset.metadata["sample_count"] == SAMPLES_PER_CLASS * EXPECTED_CLASS_COUNT
    assert dataset.metadata["class_distance"] == CLASS_DISTANCE
    assert dataset.metadata["noise_std"] == NOISE_STD_ZERO
    assert dataset.metadata["seed"] == SEED


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SyntheticBinaryClassificationConfig(samples_per_class=0),
            "samples_per_class must be greater than 0",
        ),
        (
            SyntheticBinaryClassificationConfig(class_distance=0.0),
            "class_distance must be greater than 0",
        ),
        (
            SyntheticBinaryClassificationConfig(noise_std=-1.0),
            "noise_std cannot be negative",
        ),
    ],
)
def test_synthetic_binary_classification_dataset_rejects_invalid_config(
    config: SyntheticBinaryClassificationConfig,
    expected_message: str,
) -> None:
    """Invalid dataset configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_synthetic_binary_classification_dataset(config)
