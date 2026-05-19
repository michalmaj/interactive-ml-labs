"""Command-line entry point for the Boosting Mistake Lab demo."""

from __future__ import annotations

from ml_lab_core import AlgorithmSnapshot

from boosting_mistake_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
)
from boosting_mistake_lab.report import (
    build_boosting_comparison_report,
    format_boosting_comparison_report,
)
from boosting_mistake_lab.trainer import BoostingTrainer, BoostingTrainerConfig
from boosting_mistake_lab.weak_learner import WeakLearnerBaseline

TRAINER_ROUND_COUNT: int = 5


def main() -> None:
    """Run the command-line report for the Boosting Mistake Lab demo."""
    axis_dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(dataset_kind=DATASET_KIND_AXIS_ALIGNED),
    )
    xor_dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(dataset_kind=DATASET_KIND_XOR),
    )

    print("Boosting Mistake Lab")
    print("CLI comparison report")
    print()

    _print_dataset_report(axis_dataset)
    print()
    _print_dataset_report(xor_dataset)


def _print_dataset_report(dataset: WeightedTrainTestDataset) -> None:
    """Print a formatted report for one dataset."""
    weak_snapshot = _fit_weak_learner(dataset)
    trainer_snapshot = _fit_boosting_trainer(dataset)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    print(format_boosting_comparison_report(report))


def _fit_weak_learner(dataset: WeightedTrainTestDataset) -> AlgorithmSnapshot:
    """Fit a weak learner baseline and return its snapshot."""
    weak_learner = WeakLearnerBaseline()

    return weak_learner.reset(dataset)


def _fit_boosting_trainer(dataset: WeightedTrainTestDataset) -> AlgorithmSnapshot:
    """Fit the multi-round boosting trainer and return its snapshot."""
    trainer = BoostingTrainer(
        BoostingTrainerConfig(round_count=TRAINER_ROUND_COUNT),
    )
    result = trainer.reset(dataset)

    return result.snapshot
