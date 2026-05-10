"""Random Forest Bagging Lab demo package."""

from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "SyntheticTrainTestDatasetConfig",
    "TrainTestDataset",
    "make_synthetic_train_test_dataset",
]
