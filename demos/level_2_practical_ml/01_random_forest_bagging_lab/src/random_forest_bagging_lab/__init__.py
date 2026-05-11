"""Random Forest Bagging Lab demo package."""

from random_forest_bagging_lab.baseline import (
    SingleTreeBaseline,
    SingleTreeBaselineConfig,
)
from random_forest_bagging_lab.bootstrap import (
    BootstrapSample,
    BootstrapSampleConfig,
    make_bootstrap_indices,
    make_bootstrap_sample,
)
from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)
from random_forest_bagging_lab.forest import (
    ForestTreeMember,
    RandomForestConfig,
    RandomForestModel,
)
from random_forest_bagging_lab.voting import (
    VotingResult,
    majority_vote,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "BootstrapSample",
    "BootstrapSampleConfig",
    "ForestTreeMember",
    "RandomForestConfig",
    "RandomForestModel",
    "SingleTreeBaseline",
    "SingleTreeBaselineConfig",
    "SyntheticTrainTestDatasetConfig",
    "TrainTestDataset",
    "VotingResult",
    "majority_vote",
    "make_bootstrap_indices",
    "make_bootstrap_sample",
    "make_synthetic_train_test_dataset",
]
