"""Decision Tree Splitter demo package."""

from decision_tree_splitter.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from decision_tree_splitter.impurity import (
    class_counts,
    class_probabilities,
    entropy_impurity,
    gini_impurity,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "SyntheticDecisionTreeDatasetConfig",
    "class_counts",
    "class_probabilities",
    "entropy_impurity",
    "gini_impurity",
    "make_synthetic_decision_tree_dataset",
]
