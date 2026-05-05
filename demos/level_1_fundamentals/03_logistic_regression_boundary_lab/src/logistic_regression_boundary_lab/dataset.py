"""Synthetic datasets for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import DatasetSnapshot
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

DEFAULT_SAMPLES_PER_CLASS: Final[int] = 70
DEFAULT_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_NOISE_STD: Final[float] = 1.0
DEFAULT_SEED: Final[int] = 42

CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1
CLASS_COUNT: Final[int] = 2
FEATURE_COUNT: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SyntheticBinaryClassificationConfig:
    """Configuration for a two-dimensional binary classification dataset.

    Attributes:
        samples_per_class: Number of generated samples for each class.
        class_distance: Distance between class centers.
        noise_std: Standard deviation of Gaussian noise around each center.
        seed: Random seed used for reproducible generation.
    """

    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS
    class_distance: float = DEFAULT_CLASS_DISTANCE
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = DEFAULT_SEED


def make_synthetic_binary_classification_dataset(
    config: SyntheticBinaryClassificationConfig | None = None,
) -> DatasetSnapshot:
    """Generate a simple 2D binary classification dataset.

    The generated dataset contains two Gaussian-like point clouds. The classes
    are mostly linearly separable when the noise level is low. This makes the
    dataset useful for introducing logistic regression and decision boundaries.

    Args:
        config: Optional dataset generation configuration.

    Returns:
        DatasetSnapshot containing 2D features, binary targets, and metadata.

    Raises:
        ValueError: If the configuration contains invalid values.
    """
    config = config or SyntheticBinaryClassificationConfig()
    _validate_config(config)

    rng = np.random.default_rng(config.seed)

    half_distance = config.class_distance / 2.0
    class_zero_center = np.array([-half_distance, 0.0], dtype=float)
    class_one_center = np.array([half_distance, 0.0], dtype=float)

    class_zero_features = _sample_class_points(
        rng=rng,
        center=class_zero_center,
        sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )
    class_one_features = _sample_class_points(
        rng=rng,
        center=class_one_center,
        sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )

    features = np.vstack([class_zero_features, class_one_features])
    targets = np.concatenate(
        [
            np.full(config.samples_per_class, CLASS_ZERO_LABEL, dtype=int),
            np.full(config.samples_per_class, CLASS_ONE_LABEL, dtype=int),
        ],
    )

    shuffled_indices = rng.permutation(features.shape[0])

    return DatasetSnapshot(
        features=features[shuffled_indices],
        targets=targets[shuffled_indices],
        feature_names=("x1", "x2"),
        target_names=("class_0", "class_1"),
        metadata={
            "dataset_type": "synthetic_binary_classification",
            "samples_per_class": config.samples_per_class,
            "sample_count": config.samples_per_class * CLASS_COUNT,
            "class_distance": config.class_distance,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
    )


def _sample_class_points(
    *,
    rng: np.random.Generator,
    center: FloatArray,
    sample_count: int,
    noise_std: float,
) -> FloatArray:
    """Sample points around one class center."""
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=(sample_count, FEATURE_COUNT),
    )

    return center + noise


def _validate_config(config: SyntheticBinaryClassificationConfig) -> None:
    """Validate synthetic binary classification dataset configuration."""
    if config.samples_per_class <= 0:
        msg = "samples_per_class must be greater than 0."
        raise ValueError(msg)

    if config.class_distance <= 0.0:
        msg = "class_distance must be greater than 0."
        raise ValueError(msg)

    if config.noise_std < 0.0:
        msg = "noise_std cannot be negative."
        raise ValueError(msg)
