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
from boosting_mistake_lab.weak_learner import (
    WeakLearnerBaseline,
    WeakLearnerConfig,
    WeakLearnerSplit,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "SyntheticWeightedDatasetConfig",
    "WeakLearnerBaseline",
    "WeakLearnerConfig",
    "WeakLearnerSplit",
    "WeightedDatasetSplit",
    "WeightedTrainTestDataset",
    "make_synthetic_weighted_dataset",
    "make_uniform_sample_weights",
]
