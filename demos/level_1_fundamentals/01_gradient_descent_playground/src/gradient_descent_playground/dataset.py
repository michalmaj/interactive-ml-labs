"""Synthetic datasets for the Gradient Descent Playground demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import DatasetSnapshot

DEFAULT_SAMPLE_COUNT: Final[int] = 80
DEFAULT_X_MIN: Final[float] = -5.0
DEFAULT_X_MAX: Final[float] = 5.0
DEFAULT_TRUE_WEIGHT: Final[float] = 2.0
DEFAULT_TRUE_BIAS: Final[float] = -1.0
DEFAULT_NOISE_STD: Final[float] = 0.8
DEFAULT_SEED: Final[int] = 42


@dataclass(frozen=True, slots=True)
class SyntheticRegressionConfig:
    """Configuration for a one-dimensional synthetic regression dataset.

    Attributes:
        sample_count: Number of generated samples.
        x_min: Minimum input value.
        x_max: Maximum input value.
        true_weight: True slope used to generate targets.
        true_bias: True intercept used to generate targets.
        noise_std: Standard deviation of Gaussian noise added to targets.
        seed: Random seed used for reproducible generation.
    """

    sample_count: int = DEFAULT_SAMPLE_COUNT
    x_min: float = DEFAULT_X_MIN
    x_max: float = DEFAULT_X_MAX
    true_weight: float = DEFAULT_TRUE_WEIGHT
    true_bias: float = DEFAULT_TRUE_BIAS
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = DEFAULT_SEED


def make_synthetic_regression_dataset(
    config: SyntheticRegressionConfig | None = None,
) -> DatasetSnapshot:
    """Generate a simple one-dimensional linear regression dataset.

    The generated dataset follows the equation:

    ```text
    y = true_weight * x + true_bias + noise
    ```

    Args:
        config: Optional dataset generation configuration.

    Returns:
        DatasetSnapshot containing input features, targets, and metadata.

    Raises:
        ValueError: If the configuration contains invalid values.
    """
    config = config or SyntheticRegressionConfig()
    _validate_config(config)

    rng = np.random.default_rng(config.seed)

    x_values = rng.uniform(
        low=config.x_min,
        high=config.x_max,
        size=config.sample_count,
    )

    noise = rng.normal(
        loc=0.0,
        scale=config.noise_std,
        size=config.sample_count,
    )

    targets = config.true_weight * x_values + config.true_bias + noise
    features = x_values.reshape(-1, 1)

    return DatasetSnapshot(
        features=features,
        targets=targets,
        feature_names=("x",),
        target_names=("y",),
        metadata={
            "dataset_type": "synthetic_linear_regression",
            "sample_count": config.sample_count,
            "x_min": config.x_min,
            "x_max": config.x_max,
            "true_weight": config.true_weight,
            "true_bias": config.true_bias,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
    )


def _validate_config(config: SyntheticRegressionConfig) -> None:
    """Validate synthetic regression dataset configuration."""
    if config.sample_count <= 0:
        msg = "sample_count must be greater than 0."
        raise ValueError(msg)

    if config.x_min >= config.x_max:
        msg = "x_min must be smaller than x_max."
        raise ValueError(msg)

    if config.noise_std < 0.0:
        msg = "noise_std cannot be negative."
        raise ValueError(msg)
