"""Synthetic train/test datasets for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import DatasetSnapshot
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type MetadataValue = int | float | str

DATASET_KIND_AXIS_ALIGNED: Final[str] = "axis_aligned"
DATASET_KIND_XOR: Final[str] = "xor"
VALID_DATASET_KINDS: Final[frozenset[str]] = frozenset(
    {
        DATASET_KIND_AXIS_ALIGNED,
        DATASET_KIND_XOR,
    },
)

DEFAULT_TRAIN_SAMPLES_PER_CLASS: Final[int] = 120
DEFAULT_TEST_SAMPLES_PER_CLASS: Final[int] = 120
DEFAULT_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_NOISE_STD: Final[float] = 0.65
DEFAULT_SEED: Final[int] = 42
DEFAULT_DATASET_KIND: Final[str] = DATASET_KIND_XOR

CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1
CLASS_COUNT: Final[int] = 2
FEATURE_COUNT: Final[int] = 2
XOR_BLOB_COUNT_PER_CLASS: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SyntheticTrainTestDatasetConfig:
    """Configuration for a synthetic train/test classification dataset.

    Attributes:
        train_samples_per_class: Number of training samples for each class.
        test_samples_per_class: Number of test samples for each class.
        class_distance: Distance between synthetic class centers.
        noise_std: Standard deviation of Gaussian noise around centers.
        seed: Random seed used for reproducible generation.
        dataset_kind: Dataset layout. Supported values are `axis_aligned` and `xor`.
    """

    train_samples_per_class: int = DEFAULT_TRAIN_SAMPLES_PER_CLASS
    test_samples_per_class: int = DEFAULT_TEST_SAMPLES_PER_CLASS
    class_distance: float = DEFAULT_CLASS_DISTANCE
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = DEFAULT_SEED
    dataset_kind: str = DEFAULT_DATASET_KIND


@dataclass(frozen=True, slots=True)
class TrainTestDataset:
    """Container holding train and test dataset snapshots.

    Attributes:
        train: Training split.
        test: Test split.
        metadata: Shared dataset metadata.
    """

    train: DatasetSnapshot
    test: DatasetSnapshot
    metadata: dict[str, MetadataValue]


def make_synthetic_train_test_dataset(
    config: SyntheticTrainTestDatasetConfig | None = None,
) -> TrainTestDataset:
    """Generate a reproducible synthetic train/test classification dataset.

    Args:
        config: Optional dataset configuration.

    Returns:
        TrainTestDataset with separate training and test snapshots.

    Raises:
        ValueError: If configuration values are invalid.
    """
    config = config or SyntheticTrainTestDatasetConfig()
    _validate_config(config)

    rng = np.random.default_rng(config.seed)

    train = _make_dataset_split(
        config=config,
        rng=rng,
        samples_per_class=config.train_samples_per_class,
        split_name="train",
    )
    test = _make_dataset_split(
        config=config,
        rng=rng,
        samples_per_class=config.test_samples_per_class,
        split_name="test",
    )

    metadata: dict[str, MetadataValue] = {
        "dataset_type": "synthetic_random_forest_classification",
        "dataset_kind": config.dataset_kind,
        "train_samples_per_class": config.train_samples_per_class,
        "test_samples_per_class": config.test_samples_per_class,
        "train_sample_count": config.train_samples_per_class * CLASS_COUNT,
        "test_sample_count": config.test_samples_per_class * CLASS_COUNT,
        "class_distance": config.class_distance,
        "noise_std": config.noise_std,
        "seed": config.seed,
    }

    return TrainTestDataset(
        train=train,
        test=test,
        metadata=metadata,
    )


def _make_dataset_split(
    *,
    config: SyntheticTrainTestDatasetConfig,
    rng: np.random.Generator,
    samples_per_class: int,
    split_name: str,
) -> DatasetSnapshot:
    """Generate one dataset split."""
    if config.dataset_kind == DATASET_KIND_AXIS_ALIGNED:
        features, targets = _make_axis_aligned_features(
            samples_per_class=samples_per_class,
            class_distance=config.class_distance,
            noise_std=config.noise_std,
            rng=rng,
        )
    elif config.dataset_kind == DATASET_KIND_XOR:
        features, targets = _make_xor_features(
            samples_per_class=samples_per_class,
            class_distance=config.class_distance,
            noise_std=config.noise_std,
            rng=rng,
        )
    else:
        msg = f"Unsupported dataset kind: {config.dataset_kind}."
        raise ValueError(msg)

    shuffled_indices = rng.permutation(features.shape[0])
    shuffled_features = features[shuffled_indices]
    shuffled_targets = targets[shuffled_indices]

    return DatasetSnapshot(
        features=shuffled_features,
        targets=shuffled_targets,
        feature_names=("x1", "x2"),
        target_names=("class_0", "class_1"),
        metadata={
            "dataset_type": "synthetic_random_forest_classification",
            "dataset_kind": config.dataset_kind,
            "split": split_name,
            "samples_per_class": samples_per_class,
            "sample_count": samples_per_class * CLASS_COUNT,
            "class_distance": config.class_distance,
            "noise_std": config.noise_std,
            "seed": config.seed,
        },
    )


def _make_axis_aligned_features(
    *,
    samples_per_class: int,
    class_distance: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    """Generate a dataset separable mostly by one vertical split."""
    half_distance = class_distance / 2.0

    class_zero_center = np.array([-half_distance, 0.0], dtype=float)
    class_one_center = np.array([half_distance, 0.0], dtype=float)

    class_zero_features = _sample_blob(
        rng=rng,
        center=class_zero_center,
        sample_count=samples_per_class,
        noise_std=noise_std,
    )
    class_one_features = _sample_blob(
        rng=rng,
        center=class_one_center,
        sample_count=samples_per_class,
        noise_std=noise_std,
    )

    features = np.vstack([class_zero_features, class_one_features])
    targets = _make_balanced_targets(samples_per_class)

    return features, targets


def _make_xor_features(
    *,
    samples_per_class: int,
    class_distance: float,
    noise_std: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    """Generate an XOR-like dataset useful for tree ensembles."""
    half_distance = class_distance / 2.0

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
        total_sample_count=samples_per_class,
        noise_std=noise_std,
    )
    class_one_features = _sample_multiple_blobs(
        rng=rng,
        centers=class_one_centers,
        total_sample_count=samples_per_class,
        noise_std=noise_std,
    )

    features = np.vstack([class_zero_features, class_one_features])
    targets = _make_balanced_targets(samples_per_class)

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


def _make_balanced_targets(samples_per_class: int) -> IntArray:
    """Create binary targets with equal class counts."""
    return np.concatenate(
        [
            np.full(samples_per_class, CLASS_ZERO_LABEL, dtype=int),
            np.full(samples_per_class, CLASS_ONE_LABEL, dtype=int),
        ],
    )


def _validate_config(config: SyntheticTrainTestDatasetConfig) -> None:
    """Validate dataset configuration."""
    if config.train_samples_per_class <= 0:
        msg = "train_samples_per_class must be greater than 0."
        raise ValueError(msg)

    if config.test_samples_per_class <= 0:
        msg = "test_samples_per_class must be greater than 0."
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
