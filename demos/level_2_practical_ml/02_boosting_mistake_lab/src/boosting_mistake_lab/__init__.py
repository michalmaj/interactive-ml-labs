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
from boosting_mistake_lab.learner_weight import (
    LearnerWeightResult,
    compute_learner_weight,
)
from boosting_mistake_lab.sample_weight_update import (
    SampleWeightUpdateResult,
    update_sample_weights,
)
from boosting_mistake_lab.weak_learner import (
    WeakLearnerBaseline,
    WeakLearnerConfig,
    WeakLearnerSplit,
)
from boosting_mistake_lab.weighted_error import (
    WeightedErrorResult,
    evaluate_weighted_predictions,
    weighted_accuracy_score,
    weighted_error_score,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "LearnerWeightResult",
    "SampleWeightUpdateResult",
    "SyntheticWeightedDatasetConfig",
    "WeakLearnerBaseline",
    "WeakLearnerConfig",
    "WeakLearnerSplit",
    "WeightedDatasetSplit",
    "WeightedErrorResult",
    "WeightedTrainTestDataset",
    "compute_learner_weight",
    "evaluate_weighted_predictions",
    "make_synthetic_weighted_dataset",
    "make_uniform_sample_weights",
    "update_sample_weights",
    "weighted_accuracy_score",
    "weighted_error_score",
]
