"""Decision Tree Splitter demo package."""

from decision_tree_splitter.challenge import (
    DecisionTreeChallenge,
    DecisionTreeChallengeConfig,
    DecisionTreeChallengeResult,
)
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
from decision_tree_splitter.manual_split import (
    ManualSplitConfig,
    ManualSplitPrototype,
)
from decision_tree_splitter.split import (
    SplitCandidate,
    SplitEvaluation,
    best_split,
    evaluate_split,
    generate_split_candidates,
)
from decision_tree_splitter.stump import (
    DecisionStump,
    DecisionStumpConfig,
    LeafPrediction,
)
from decision_tree_splitter.tree import (
    DecisionTreeConfig,
    DecisionTreeNode,
    RecursiveDecisionTree,
)

__all__ = [
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "DecisionStump",
    "DecisionStumpConfig",
    "DecisionTreeChallenge",
    "DecisionTreeChallengeConfig",
    "DecisionTreeChallengeResult",
    "DecisionTreeConfig",
    "DecisionTreeNode",
    "LeafPrediction",
    "ManualSplitConfig",
    "ManualSplitPrototype",
    "RecursiveDecisionTree",
    "SplitCandidate",
    "SplitEvaluation",
    "SyntheticDecisionTreeDatasetConfig",
    "best_split",
    "class_counts",
    "class_probabilities",
    "entropy_impurity",
    "evaluate_split",
    "generate_split_candidates",
    "gini_impurity",
    "make_synthetic_decision_tree_dataset",
]
