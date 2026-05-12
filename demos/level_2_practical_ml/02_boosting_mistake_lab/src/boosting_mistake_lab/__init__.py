"""Boosting Mistake Lab demo package."""

from boosting_mistake_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeightedDatasetSplit,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
    make_uniform_sample_weights,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "SyntheticWeightedDatasetConfig",
    "WeightedDatasetSplit",
    "WeightedTrainTestDataset",
    "make_synthetic_weighted_dataset",
    "make_uniform_sample_weights",
]
