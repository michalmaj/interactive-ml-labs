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
from random_forest_bagging_lab.challenge import (
    RandomForestChallenge,
    RandomForestChallengeConfig,
    RandomForestChallengeResult,
)
from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)
from random_forest_bagging_lab.explanation import (
    build_bottom_panel_explanation,
    build_challenge_target_text,
    build_confidence_view_explanation,
    build_gap_limit_text,
    build_tree_limit_text,
)
from random_forest_bagging_lab.forest import (
    ForestTreeMember,
    RandomForestConfig,
    RandomForestModel,
)
from random_forest_bagging_lab.report import (
    ModelComparisonReport,
    ModelReportMetrics,
    build_model_comparison_report,
    format_model_comparison_report,
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
    "ModelComparisonReport",
    "ModelReportMetrics",
    "RandomForestChallenge",
    "RandomForestChallengeConfig",
    "RandomForestChallengeResult",
    "RandomForestConfig",
    "RandomForestModel",
    "SingleTreeBaseline",
    "SingleTreeBaselineConfig",
    "SyntheticTrainTestDatasetConfig",
    "TrainTestDataset",
    "VotingResult",
    "build_bottom_panel_explanation",
    "build_challenge_target_text",
    "build_confidence_view_explanation",
    "build_gap_limit_text",
    "build_model_comparison_report",
    "build_tree_limit_text",
    "format_model_comparison_report",
    "majority_vote",
    "make_bootstrap_indices",
    "make_bootstrap_sample",
    "make_synthetic_train_test_dataset",
]
