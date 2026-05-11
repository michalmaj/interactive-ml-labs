"""Bootstrap sampling utilities for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import DatasetSnapshot
from numpy.typing import NDArray

type IntArray = NDArray[np.int_]
type MetadataValue = int | float | str

DEFAULT_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 1.0
DEFAULT_BOOTSTRAP_SEED: Final[int] = 42

MIN_SAMPLE_COUNT: Final[int] = 1
MIN_SAMPLE_RATIO: Final[float] = 0.0
MAX_SAMPLE_RATIO: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class BootstrapSampleConfig:
    """Configuration for one bootstrap sample.

    Attributes:
        sample_ratio: Fraction of the source dataset size to draw with replacement.
            A value of 1.0 means that the bootstrap sample has the same size as
            the source dataset.
        seed: Random seed used for reproducible bootstrap sampling.
    """

    sample_ratio: float = DEFAULT_BOOTSTRAP_SAMPLE_RATIO
    seed: int = DEFAULT_BOOTSTRAP_SEED


@dataclass(frozen=True, slots=True)
class BootstrapSample:
    """Result of bootstrap sampling.

    Attributes:
        dataset: Dataset snapshot containing bootstrapped features and targets.
        sample_indices: Source dataset indices drawn with replacement.
        oob_indices: Out-of-bag indices, not drawn into this bootstrap sample.
        unique_sample_count: Number of unique source samples used by the bootstrap sample.
    """

    dataset: DatasetSnapshot
    sample_indices: IntArray
    oob_indices: IntArray
    unique_sample_count: int


def make_bootstrap_sample(
    dataset: DatasetSnapshot,
    config: BootstrapSampleConfig | None = None,
) -> BootstrapSample:
    """Create one bootstrap sample from a labeled dataset.

    Bootstrap sampling draws source indices with replacement. Some source
    samples may be selected many times, while others may not be selected at all.

    Samples not selected into the bootstrap dataset are called out-of-bag
    samples and can later be used for additional model evaluation intuition.

    Args:
        dataset: Source labeled dataset.
        config: Optional bootstrap configuration.

    Returns:
        BootstrapSample with bootstrapped dataset, sampled indices, and OOB indices.

    Raises:
        ValueError: If the dataset or configuration is invalid.
    """
    config = config or BootstrapSampleConfig()
    _validate_config(config)

    features = np.asarray(dataset.features)
    targets = _extract_targets(dataset)

    _validate_dataset_arrays(features, targets)

    sample_indices, oob_indices = make_bootstrap_indices(
        sample_count=features.shape[0],
        config=config,
    )

    bootstrap_dataset = DatasetSnapshot(
        features=features[sample_indices],
        targets=targets[sample_indices],
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        metadata=_build_bootstrap_metadata(
            source_sample_count=features.shape[0],
            sample_indices=sample_indices,
            oob_indices=oob_indices,
            config=config,
        ),
    )

    return BootstrapSample(
        dataset=bootstrap_dataset,
        sample_indices=sample_indices,
        oob_indices=oob_indices,
        unique_sample_count=int(np.unique(sample_indices).shape[0]),
    )


def make_bootstrap_indices(
    *,
    sample_count: int,
    config: BootstrapSampleConfig | None = None,
) -> tuple[IntArray, IntArray]:
    """Draw bootstrap source indices and compute out-of-bag indices.

    Args:
        sample_count: Number of samples in the source dataset.
        config: Optional bootstrap configuration.

    Returns:
        Pair `(sample_indices, oob_indices)`.

    Raises:
        ValueError: If sample_count or configuration values are invalid.
    """
    config = config or BootstrapSampleConfig()
    _validate_config(config)

    if sample_count < MIN_SAMPLE_COUNT:
        msg = "sample_count must be greater than 0."
        raise ValueError(msg)

    draw_count = _resolve_draw_count(
        sample_count=sample_count,
        sample_ratio=config.sample_ratio,
    )
    rng = np.random.default_rng(config.seed)

    sample_indices = rng.integers(
        low=0,
        high=sample_count,
        size=draw_count,
        dtype=int,
    )
    oob_indices = _compute_oob_indices(
        sample_count=sample_count,
        sample_indices=sample_indices,
    )

    return sample_indices, oob_indices


def _extract_targets(dataset: DatasetSnapshot) -> IntArray:
    """Extract target labels from a dataset snapshot."""
    if dataset.targets is None:
        msg = "Dataset targets are required for bootstrap sampling."
        raise ValueError(msg)

    targets = np.asarray(dataset.targets)

    if not np.issubdtype(targets.dtype, np.integer):
        msg = "Dataset targets must contain integers."
        raise ValueError(msg)

    return targets.astype(int)


def _validate_dataset_arrays(features: NDArray[np.generic], targets: IntArray) -> None:
    """Validate source dataset arrays."""
    if features.ndim == 0:
        msg = "Dataset features must contain at least one sample."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "Dataset features cannot be empty."
        raise ValueError(msg)

    if targets.ndim != 1:
        msg = "Dataset targets must be a one-dimensional array."
        raise ValueError(msg)

    if features.shape[0] != targets.shape[0]:
        msg = (
            "Dataset features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)


def _resolve_draw_count(*, sample_count: int, sample_ratio: float) -> int:
    """Resolve number of bootstrap draws."""
    return max(MIN_SAMPLE_COUNT, round(sample_count * sample_ratio))


def _compute_oob_indices(*, sample_count: int, sample_indices: IntArray) -> IntArray:
    """Compute out-of-bag indices for one bootstrap draw."""
    selected_mask = np.zeros(sample_count, dtype=bool)
    selected_mask[np.unique(sample_indices)] = True

    return np.flatnonzero(~selected_mask).astype(int)


def _build_bootstrap_metadata(
    *,
    source_sample_count: int,
    sample_indices: IntArray,
    oob_indices: IntArray,
    config: BootstrapSampleConfig,
) -> dict[str, MetadataValue]:
    """Build metadata for a bootstrap dataset snapshot."""
    unique_sample_count = int(np.unique(sample_indices).shape[0])
    draw_count = int(sample_indices.shape[0])
    oob_sample_count = int(oob_indices.shape[0])

    return {
        "dataset_type": "bootstrap_sample",
        "source_sample_count": source_sample_count,
        "draw_count": draw_count,
        "sample_ratio": config.sample_ratio,
        "unique_sample_count": unique_sample_count,
        "oob_sample_count": oob_sample_count,
        "oob_fraction": oob_sample_count / source_sample_count,
        "seed": config.seed,
    }


def _validate_config(config: BootstrapSampleConfig) -> None:
    """Validate bootstrap configuration."""
    if not MIN_SAMPLE_RATIO < config.sample_ratio <= MAX_SAMPLE_RATIO:
        msg = "sample_ratio must be in the range (0, 1]."
        raise ValueError(msg)
