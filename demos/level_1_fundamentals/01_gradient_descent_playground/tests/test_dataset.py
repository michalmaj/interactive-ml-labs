"""Tests for synthetic regression dataset generation."""

import numpy as np
import pytest
from gradient_descent_playground import (
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)

SAMPLE_COUNT: int = 12
X_MIN: float = -2.0
X_MAX: float = 3.0
TRUE_WEIGHT: float = 2.5
TRUE_BIAS: float = -0.7
NOISE_STD_ZERO: float = 0.0
SEED: int = 123


def test_synthetic_regression_dataset_has_expected_shape() -> None:
    """Generated features and targets should have expected shapes."""
    config = SyntheticRegressionConfig(sample_count=SAMPLE_COUNT)
    dataset = make_synthetic_regression_dataset(config)

    features = np.asarray(dataset.features)
    targets = np.asarray(dataset.targets)

    assert features.shape == (SAMPLE_COUNT, 1)
    assert targets.shape == (SAMPLE_COUNT,)


def test_synthetic_regression_dataset_is_reproducible_for_same_seed() -> None:
    """The same seed should generate exactly the same dataset."""
    config = SyntheticRegressionConfig(sample_count=SAMPLE_COUNT, seed=SEED)

    first = make_synthetic_regression_dataset(config)
    second = make_synthetic_regression_dataset(config)

    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_allclose(first.targets, second.targets)


def test_synthetic_regression_dataset_uses_linear_relation_without_noise() -> None:
    """Without noise, targets should follow the exact linear relation."""
    config = SyntheticRegressionConfig(
        sample_count=SAMPLE_COUNT,
        true_weight=TRUE_WEIGHT,
        true_bias=TRUE_BIAS,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
    )

    dataset = make_synthetic_regression_dataset(config)
    features = np.asarray(dataset.features).ravel()
    targets = np.asarray(dataset.targets)

    expected_targets = TRUE_WEIGHT * features + TRUE_BIAS

    np.testing.assert_allclose(targets, expected_targets)


def test_synthetic_regression_dataset_stores_metadata() -> None:
    """Generated dataset should expose useful metadata for demos."""
    config = SyntheticRegressionConfig(
        sample_count=SAMPLE_COUNT,
        x_min=X_MIN,
        x_max=X_MAX,
        true_weight=TRUE_WEIGHT,
        true_bias=TRUE_BIAS,
        noise_std=NOISE_STD_ZERO,
        seed=SEED,
    )

    dataset = make_synthetic_regression_dataset(config)

    assert dataset.feature_names == ("x",)
    assert dataset.target_names == ("y",)
    assert dataset.metadata["dataset_type"] == "synthetic_linear_regression"
    assert dataset.metadata["sample_count"] == SAMPLE_COUNT
    assert dataset.metadata["x_min"] == X_MIN
    assert dataset.metadata["x_max"] == X_MAX
    assert dataset.metadata["true_weight"] == TRUE_WEIGHT
    assert dataset.metadata["true_bias"] == TRUE_BIAS
    assert dataset.metadata["noise_std"] == NOISE_STD_ZERO
    assert dataset.metadata["seed"] == SEED


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            SyntheticRegressionConfig(sample_count=0),
            "sample_count must be greater than 0",
        ),
        (
            SyntheticRegressionConfig(x_min=1.0, x_max=1.0),
            "x_min must be smaller than x_max",
        ),
        (
            SyntheticRegressionConfig(noise_std=-1.0),
            "noise_std cannot be negative",
        ),
    ],
)
def test_synthetic_regression_dataset_rejects_invalid_config(
    config: SyntheticRegressionConfig,
    expected_message: str,
) -> None:
    """Invalid dataset configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        make_synthetic_regression_dataset(config)
