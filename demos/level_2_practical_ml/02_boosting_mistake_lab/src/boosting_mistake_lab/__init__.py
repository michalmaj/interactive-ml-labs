"""Boosting Mistake Lab demo package."""

from boosting_mistake_lab.boosted_prediction import (
    BoostedPredictionResult,
    boosted_accuracy_score,
    predict_boosted_ensemble,
)
from boosting_mistake_lab.boosting_round import (
    BoostingRoundConfig,
    BoostingRoundResult,
    make_next_round_dataset,
    run_boosting_round,
)
from boosting_mistake_lab.challenge import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    BoostingChallengeConfig,
    BoostingChallengeResult,
    evaluate_boosting_challenge,
)
from boosting_mistake_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeightedDatasetSplit,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
    make_uniform_sample_weights,
)
from boosting_mistake_lab.explanation import (
    BoostingExplanation,
    build_boosting_explanation,
)
from boosting_mistake_lab.learner_weight import (
    LearnerWeightResult,
    compute_learner_weight,
)
from boosting_mistake_lab.report import (
    BOOSTED_ENSEMBLE_NAME,
    WEAK_LEARNER_NAME,
    BoostedEnsembleReportMetrics,
    BoostingComparisonReport,
    WeakLearnerReportMetrics,
    build_boosting_comparison_report,
    format_boosting_comparison_report,
)
from boosting_mistake_lab.sample_weight_update import (
    SampleWeightUpdateResult,
    update_sample_weights,
)
from boosting_mistake_lab.staged_history import (
    StagedBoostingHistory,
    build_staged_boosting_history,
)
from boosting_mistake_lab.trainer import (
    BoostingTrainer,
    BoostingTrainerConfig,
    BoostingTrainerResult,
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
    "BOOSTED_ENSEMBLE_NAME",
    "DATASET_KIND_AXIS_ALIGNED",
    "DATASET_KIND_XOR",
    "STATUS_FAILED",
    "STATUS_SUCCESS",
    "WEAK_LEARNER_NAME",
    "BoostedEnsembleReportMetrics",
    "BoostedPredictionResult",
    "BoostingChallengeConfig",
    "BoostingChallengeResult",
    "BoostingComparisonReport",
    "BoostingExplanation",
    "BoostingRoundConfig",
    "BoostingRoundResult",
    "BoostingTrainer",
    "BoostingTrainerConfig",
    "BoostingTrainerResult",
    "LearnerWeightResult",
    "SampleWeightUpdateResult",
    "StagedBoostingHistory",
    "SyntheticWeightedDatasetConfig",
    "WeakLearnerBaseline",
    "WeakLearnerConfig",
    "WeakLearnerReportMetrics",
    "WeakLearnerSplit",
    "WeightedDatasetSplit",
    "WeightedErrorResult",
    "WeightedTrainTestDataset",
    "boosted_accuracy_score",
    "build_boosting_comparison_report",
    "build_boosting_explanation",
    "build_staged_boosting_history",
    "compute_learner_weight",
    "evaluate_boosting_challenge",
    "evaluate_weighted_predictions",
    "format_boosting_comparison_report",
    "make_next_round_dataset",
    "make_synthetic_weighted_dataset",
    "make_uniform_sample_weights",
    "predict_boosted_ensemble",
    "run_boosting_round",
    "update_sample_weights",
    "weighted_accuracy_score",
    "weighted_error_score",
]
