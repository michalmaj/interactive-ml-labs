"""Tests for bootstrap sampling utilities."""

import numpy as np
import pytest
from ml_lab_core import DatasetSnapshot
from random_forest_bagging_lab import (
    BootstrapSampleConfig,
    SyntheticTrainTestDatasetConfig,
    make_bootstrap_indices,
    make_bootstrap_sample,
    make_synthetic_train_test_dataset,
)

TRAIN_SAMPLES_PER_CLASS: int = 20
TEST_SAMPLES_PER_CLASS: int = 8
SOURCE_SAMPLE_COUNT: int = TRAIN_SAMPLES_PER_CLASS * 2
HALF_SAMPLE_RATIO: float = 0.5
HALF_DRAW_COUNT: int = int(SOURCE_SAMPLE_COUNT * HALF_SAMPLE_RATIO)
SEED: int = 123
OTHER_SEED: int = 124


def _train_dataset() -> DatasetSnapshot:
    """Create a deterministic training dataset for bootstrap tests."""
    train_test_dataset = make_synthetic_train_test_dataset(
        SyntheticTrainTestDatasetConfig(
            train_samples_per_class=TRAIN_SAMPLES_PER_CLASS,
            test_samples_per_class=TEST_SAMPLES_PER_CLASS,
            seed=SEED,
        ),
    )

    return train_test_dataset.train


def test_make_bootstrap_indices_draws_expected_count() -> None:
    """Bootstrap index draw count should follow sample_ratio."""
    sample_indices, oob_indices = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=BootstrapSampleConfig(
            sample_ratio=HALF_SAMPLE_RATIO,
            seed=SEED,
        ),
    )

    assert sample_indices.shape == (HALF_DRAW_COUNT,)
    assert np.all(sample_indices >= 0)
    assert np.all(sample_indices < SOURCE_SAMPLE_COUNT)
    assert np.all(oob_indices >= 0)
    assert np.all(oob_indices < SOURCE_SAMPLE_COUNT)


def test_make_bootstrap_indices_is_reproducible_for_same_seed() -> None:
    """The same seed should produce the same bootstrap indices."""
    config = BootstrapSampleConfig(seed=SEED)

    first_indices, first_oob = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=config,
    )
    second_indices, second_oob = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=config,
    )

    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_oob, second_oob)


def test_make_bootstrap_indices_changes_for_different_seed() -> None:
    """Different seeds should usually produce different bootstrap draws."""
    first_indices, _ = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=BootstrapSampleConfig(seed=SEED),
    )
    second_indices, _ = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=BootstrapSampleConfig(seed=OTHER_SEED),
    )

    assert not np.array_equal(first_indices, second_indices)


def test_make_bootstrap_indices_contains_duplicates() -> None:
    """Bootstrap sampling should draw with replacement."""
    sample_indices, _ = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=BootstrapSampleConfig(seed=SEED),
    )

    unique_sample_count = np.unique(sample_indices).shape[0]

    assert unique_sample_count < sample_indices.shape[0]


def test_make_bootstrap_indices_computes_oob_indices() -> None:
    """OOB indices should be exactly samples not selected into the bootstrap draw."""
    sample_indices, oob_indices = make_bootstrap_indices(
        sample_count=SOURCE_SAMPLE_COUNT,
        config=BootstrapSampleConfig(seed=SEED),
    )

    selected_indices = set(sample_indices.tolist())
    oob_index_set = set(oob_indices.tolist())
    all_indices = set(range(SOURCE_SAMPLE_COUNT))

    assert oob_index_set == all_indices - selected_indices
    assert selected_indices.isdisjoint(oob_index_set)


def test_make_bootstrap_sample_returns_dataset_snapshot() -> None:
    """Bootstrap sample should contain bootstrapped features and targets."""
    source_dataset = _train_dataset()

    bootstrap = make_bootstrap_sample(
        source_dataset,
        BootstrapSampleConfig(seed=SEED),
    )

    source_features = np.asarray(source_dataset.features)
    source_targets = np.asarray(source_dataset.targets)
    bootstrap_features = np.asarray(bootstrap.dataset.features)
    bootstrap_targets = np.asarray(bootstrap.dataset.targets)

    assert bootstrap_features.shape == source_features.shape
    assert bootstrap_targets.shape == source_targets.shape
    np.testing.assert_allclose(bootstrap_features, source_features[bootstrap.sample_indices])
    np.testing.assert_array_equal(bootstrap_targets, source_targets[bootstrap.sample_indices])


def test_make_bootstrap_sample_stores_metadata() -> None:
    """Bootstrap sample should expose useful metadata."""
    source_dataset = _train_dataset()

    bootstrap = make_bootstrap_sample(
        source_dataset,
        BootstrapSampleConfig(seed=SEED),
    )

    assert bootstrap.dataset.metadata["dataset_type"] == "bootstrap_sample"
    assert bootstrap.dataset.metadata["source_sample_count"] == SOURCE_SAMPLE_COUNT
    assert bootstrap.dataset.metadata["draw_count"] == SOURCE_SAMPLE_COUNT
    assert bootstrap.dataset.metadata["sample_ratio"] == 1.0
    assert bootstrap.dataset.metadata["unique_sample_count"] == bootstrap.unique_sample_count
    assert bootstrap.dataset.metadata["oob_sample_count"] == bootstrap.oob_indices.shape[0]
    assert bootstrap.dataset.metadata["seed"] == SEED


def test_make_bootstrap_sample_preserves_dataset_names() -> None:
    """Bootstrap sample should preserve feature and target names."""
    source_dataset = _train_dataset()

    bootstrap = make_bootstrap_sample(source_dataset, BootstrapSampleConfig(seed=SEED))

    assert bootstrap.dataset.feature_names == source_dataset.feature_names
    assert bootstrap.dataset.target_names == source_dataset.target_names


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            BootstrapSampleConfig(sample_ratio=0.0),
            "sample_ratio",
        ),
        (
            BootstrapSampleConfig(sample_ratio=1.1),
            "sample_ratio",
        ),
    ],
)
def test_bootstrap_rejects_invalid_config(
    config: BootstrapSampleConfig,
    expected_message: str,
) -> None:
    """Invalid bootstrap configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_bootstrap_indices(
            sample_count=SOURCE_SAMPLE_COUNT,
            config=config,
        )


def test_bootstrap_indices_rejects_invalid_sample_count() -> None:
    """Bootstrap indices require at least one source sample."""
    with pytest.raises(ValueError, match="sample_count"):
        make_bootstrap_indices(sample_count=0)


@pytest.mark.parametrize(
    "dataset, expected_message",
    [
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=None,
            ),
            "targets are required",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([0.0, 1.0]),
            ),
            "targets must contain integers",
        ),
        (
            DatasetSnapshot(
                features=np.empty((0, 2)),
                targets=np.array([], dtype=int),
            ),
            "features cannot be empty",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([[0], [1]]),
            ),
            "one-dimensional",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([0]),
            ),
            "same number of samples",
        ),
    ],
)
def test_make_bootstrap_sample_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid source datasets should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_bootstrap_sample(dataset)
