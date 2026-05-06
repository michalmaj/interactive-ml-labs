"""Synthetic datasets for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import DatasetSnapshot
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

DATASET_KIND_AXIS_ALIGNED: Final[str] = "axis_aligned"
DATASET_KIND_XOR: Final[str] = "xor"
VALID_DATASET_KINDS: Final[frozenset[str]] = frozenset(
    {
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    },
)

DEFAULT_SAMPLES_PER_CLASS: Final[int] = 80
DEFAULT_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_NOISE_STD: Final[float] = 0.45
DEFAULT_SEED: Final[int] = 42
DEFAULT_DATASET_KIND: Final[str] = DATASET_KIND_AXIS_ALIGNED

CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1
CLASS_COUNT: Final[int] = 2
FEATURE_COUNT: Final[int] = 2
BLOB_COUNT_PER_CLASS_XOR: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SyntheticDecisionTreeDatasetConfig:
    """Configuration for a two-dimensional decision-tree dataset.

    Attributes:
        samples_per_class: Number of generated samples for each class.
        class_distance: Distance between class centers.
        noise_std: Standard deviation of Gaussian noise around each center.
        seed: Random seed used for reproducible generation.
        dataset_kind: Dataset layout. Supported values are `axis_aligned` and `xor`.
    """

    samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS
    class_distance: float = DEFAULT_CLASS_DISTANCE
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = DEFAULT_SEED
    dataset_kind: str = DEFAULT_DATASET_KIND


def make_synthetic_decision_tree_dataset(
    config: SyntheticDecisionTreeDatasetConfig | None = None,
) -> DatasetSnapshot:
    """Generate a simple 2D binary classification dataset.

    The `axis_aligned` dataset is intentionally easy for a decision stump:
    a single vertical split can separate the two classes when noise is low.

    The `xor` dataset is intentionally harder: one split is not enough, so it
    will be useful later for demonstrating tree depth and recursive splitting.

    Args:
        config: Optional dataset generation configuration.

    Returns:
        DatasetSnapshot containing 2D features, binary targets, and metadata.

    Raises:
        ValueError: If the configuration contains invalid values.
    """
    config = config or SyntheticDecisionTreeDatasetConfig()
    _validate_config(config)

    rng = np.random.default_rng(config.seed)

    if config.dataset_kind == DATASET_KIND_AXIS_ALIGNED:
        features, targets = _make_axis_aligned_dataset(config, rng)
    elif config.dataset_kind == DATASET_KIND_XOR:
        features, targets = _make_xor_dataset(config, rng)
    else:
        msg = f"Unsupported dataset kind: {config.dataset_kind}."
        raise ValueError(msg)

    shuffled_indices = rng.permutation(features.shape[0])

    return DatasetSnapshot(
        features=features[shuffled_indices],
        targets=targets[shuffled_indices],
        feature_names=("x1", "x2"),
        target_names=("class_0", "class_1"),
        metadata={
            "dataset_type": "synthetic_decision_tree_classification",
            "dataset_kind": config.dataset_kind,
            "samples_per_class": config.samples_per_class,
            "sample_count": config.samples_per_class * CLASS_COUNT,
            "class_distance": config.class_distance,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
    )


def _make_axis_aligned_dataset(
    config: SyntheticDecisionTreeDatasetConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.int_]]:
    """Generate two classes separable by one vertical split."""
    half_distance = config.class_distance / 2.0

    class_zero_center = np.array([-half_distance, 0.0], dtype=float)
    class_one_center = np.array([half_distance, 0.0], dtype=float)

    class_zero_features = _sample_blob(
        rng=rng,
        center=class_zero_center,
        sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )
    class_one_features = _sample_blob(
        rng=rng,
        center=class_one_center,
        sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )

    features = np.vstack([class_zero_features, class_one_features])
    targets = _make_balanced_targets(config.samples_per_class)

    return features, targets


def _make_xor_dataset(
    config: SyntheticDecisionTreeDatasetConfig,
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.int_]]:
    """Generate an XOR-like dataset requiring multiple splits."""
    half_distance = config.class_distance / 2.0

    class_zero_centers = (
        np.array([-half_distance, -half_distance], dtype=float),
        np.array([half_distance, half_distance], dtype=float),
    )
    class_one_centers = (
        np.array([-half_distance, half_distance], dtype=float),
        np.array([half_distance, -half_distance], dtype=float),
    )

    class_zero_features = _sample_multiple_blobs(
        rng=rng,
        centers=class_zero_centers,
        total_sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )
    class_one_features = _sample_multiple_blobs(
        rng=rng,
        centers=class_one_centers,
        total_sample_count=config.samples_per_class,
        noise_std=config.noise_std,
    )

    features = np.vstack([class_zero_features, class_one_features])
    targets = _make_balanced_targets(config.samples_per_class)

    return features, targets


def _sample_multiple_blobs(
    *,
    rng: np.random.Generator,
    centers: tuple[FloatArray, ...],
    total_sample_count: int,
    noise_std: float,
) -> FloatArray:
    """Sample one class from multiple Gaussian-like blobs."""
    sample_counts = _split_count_across_blobs(
        total_sample_count=total_sample_count,
        blob_count=len(centers),
    )

    blobs = [
        _sample_blob(
            rng=rng,
            center=center,
            sample_count=sample_count,
            noise_std=noise_std,
        )
        for center, sample_count in zip(centers, sample_counts, strict=True)
    ]

    return np.vstack(blobs)


def _sample_blob(
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


def _split_count_across_blobs(*, total_sample_count: int, blob_count: int) -> tuple[int, ...]:
    """Split a total sample count across multiple blobs."""
    base_count = total_sample_count // blob_count
    remainder = total_sample_count % blob_count

    return tuple(base_count + (1 if index < remainder else 0) for index in range(blob_count))


def _make_balanced_targets(samples_per_class: int) -> NDArray[np.int_]:
    """Create binary targets with equal class counts."""
    return np.concatenate(
        [
            np.full(samples_per_class, CLASS_ZERO_LABEL, dtype=int),
            np.full(samples_per_class, CLASS_ONE_LABEL, dtype=int),
        ],
    )


def _validate_config(config: SyntheticDecisionTreeDatasetConfig) -> None:
    """Validate synthetic decision-tree dataset configuration."""
    if config.samples_per_class <= 0:
        msg = "samples_per_class must be greater than 0."
        raise ValueError(msg)

    if config.class_distance <= 0.0:
        msg = "class_distance must be greater than 0."
        raise ValueError(msg)

    if config.noise_std < 0.0:
        msg = "noise_std cannot be negative."
        raise ValueError(msg)

    if config.dataset_kind not in VALID_DATASET_KINDS:
        msg = f"dataset_kind must be one of: {', '.join(sorted(VALID_DATASET_KINDS))}."
        raise ValueError(msg)
