"""CLI comparison reports for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

from boosting_mistake_lab.dataset import WeightedTrainTestDataset

WEAK_LEARNER_NAME: Final[str] = "weak_learner"
BOOSTED_ENSEMBLE_NAME: Final[str] = "boosted_ensemble"


@dataclass(frozen=True, slots=True)
class WeakLearnerReportMetrics:
    """Metrics describing the first weak learner baseline.

    Attributes:
        name: Model name.
        train_accuracy: Weak learner train accuracy.
        test_accuracy: Weak learner test accuracy.
        generalization_gap: Train-test accuracy gap.
        weighted_train_error: Weighted train error.
        learner_weight: Learner alpha value.
        feature_index: Selected split feature index.
        threshold: Selected split threshold.
    """

    name: str
    train_accuracy: float
    test_accuracy: float
    generalization_gap: float
    weighted_train_error: float
    learner_weight: float
    feature_index: int
    threshold: float


@dataclass(frozen=True, slots=True)
class BoostedEnsembleReportMetrics:
    """Metrics describing the final boosted ensemble.

    Attributes:
        name: Model name.
        train_accuracy: Boosted train accuracy.
        test_accuracy: Boosted test accuracy.
        generalization_gap: Train-test accuracy gap.
        mean_test_confidence: Mean confidence on test split.
        round_count: Number of boosting rounds.
        best_staged_test_accuracy: Best staged test accuracy.
        best_staged_round_index: Round where best staged test accuracy was reached.
    """

    name: str
    train_accuracy: float
    test_accuracy: float
    generalization_gap: float
    mean_test_confidence: float
    round_count: int
    best_staged_test_accuracy: float
    best_staged_round_index: int


@dataclass(frozen=True, slots=True)
class BoostingComparisonReport:
    """Comparison between weak learner baseline and boosted ensemble.

    Attributes:
        dataset_kind: Dataset kind used in the report.
        train_sample_count: Number of training samples.
        test_sample_count: Number of test samples.
        weak_learner: Weak learner metrics.
        boosted_ensemble: Boosted ensemble metrics.
        train_accuracy_delta: Boosted train accuracy minus weak learner train accuracy.
        test_accuracy_delta: Boosted test accuracy minus weak learner test accuracy.
        winner: Winner by test accuracy.
        summary: Short text summary.
    """

    dataset_kind: str
    train_sample_count: int
    test_sample_count: int
    weak_learner: WeakLearnerReportMetrics
    boosted_ensemble: BoostedEnsembleReportMetrics
    train_accuracy_delta: float
    test_accuracy_delta: float
    winner: str
    summary: str


def build_boosting_comparison_report(
    *,
    dataset: WeightedTrainTestDataset,
    weak_snapshot: AlgorithmSnapshot,
    trainer_snapshot: AlgorithmSnapshot,
) -> BoostingComparisonReport:
    """Build a comparison report for CLI output.

    Args:
        dataset: Weighted train/test dataset.
        weak_snapshot: Snapshot from the first weak learner baseline.
        trainer_snapshot: Snapshot from the multi-round boosting trainer.

    Returns:
        BoostingComparisonReport with baseline and ensemble metrics.
    """
    weak_learner = _build_weak_learner_metrics(weak_snapshot)
    boosted_ensemble = _build_boosted_ensemble_metrics(trainer_snapshot)

    train_accuracy_delta = boosted_ensemble.train_accuracy - weak_learner.train_accuracy
    test_accuracy_delta = boosted_ensemble.test_accuracy - weak_learner.test_accuracy
    winner = _winner_by_test_accuracy(
        weak_test_accuracy=weak_learner.test_accuracy,
        boosted_test_accuracy=boosted_ensemble.test_accuracy,
    )

    return BoostingComparisonReport(
        dataset_kind=str(dataset.metadata["dataset_kind"]),
        train_sample_count=int(dataset.metadata["train_sample_count"]),
        test_sample_count=int(dataset.metadata["test_sample_count"]),
        weak_learner=weak_learner,
        boosted_ensemble=boosted_ensemble,
        train_accuracy_delta=train_accuracy_delta,
        test_accuracy_delta=test_accuracy_delta,
        winner=winner,
        summary=_build_summary(
            dataset_kind=str(dataset.metadata["dataset_kind"]),
            winner=winner,
            test_accuracy_delta=test_accuracy_delta,
            best_staged_round_index=boosted_ensemble.best_staged_round_index,
        ),
    )


def format_boosting_comparison_report(report: BoostingComparisonReport) -> str:
    """Format a boosting comparison report for CLI output.

    Args:
        report: Boosting comparison report.

    Returns:
        Human-readable multiline report.
    """
    lines = [
        f"Dataset: {report.dataset_kind}",
        f"Train samples: {report.train_sample_count}",
        f"Test samples: {report.test_sample_count}",
        "",
        "Weak learner baseline:",
        f"  train accuracy:        {report.weak_learner.train_accuracy:.3f}",
        f"  test accuracy:         {report.weak_learner.test_accuracy:.3f}",
        f"  generalization gap:    {report.weak_learner.generalization_gap:.3f}",
        f"  weighted train error:  {report.weak_learner.weighted_train_error:.3f}",
        f"  learner alpha:         {report.weak_learner.learner_weight:.3f}",
        (
            "  split:                 "
            f"x{report.weak_learner.feature_index + 1} "
            f"<= {report.weak_learner.threshold:.3f}"
        ),
        "",
        "Boosted ensemble:",
        f"  rounds:                {report.boosted_ensemble.round_count}",
        f"  train accuracy:        {report.boosted_ensemble.train_accuracy:.3f}",
        f"  test accuracy:         {report.boosted_ensemble.test_accuracy:.3f}",
        f"  generalization gap:    {report.boosted_ensemble.generalization_gap:.3f}",
        f"  mean test confidence:  {report.boosted_ensemble.mean_test_confidence:.3f}",
        (
            "  best staged test acc:  "
            f"{report.boosted_ensemble.best_staged_test_accuracy:.3f} "
            f"at round {report.boosted_ensemble.best_staged_round_index}"
        ),
        "",
        "Comparison:",
        f"  train accuracy delta:  {report.train_accuracy_delta:+.3f}",
        f"  test accuracy delta:   {report.test_accuracy_delta:+.3f}",
        f"  winner by test acc:    {report.winner}",
        "",
        report.summary,
    ]

    return "\n".join(lines)


def _build_weak_learner_metrics(snapshot: AlgorithmSnapshot) -> WeakLearnerReportMetrics:
    """Build weak learner report metrics from a snapshot."""
    train_accuracy = float(snapshot.metrics["train_accuracy"])
    test_accuracy = float(snapshot.metrics["test_accuracy"])

    return WeakLearnerReportMetrics(
        name=WEAK_LEARNER_NAME,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        generalization_gap=train_accuracy - test_accuracy,
        weighted_train_error=float(snapshot.metrics["weighted_train_error"]),
        learner_weight=float(snapshot.metrics["learner_weight"]),
        feature_index=int(snapshot.metrics["feature_index"]),
        threshold=float(snapshot.metrics["threshold"]),
    )


def _build_boosted_ensemble_metrics(
    snapshot: AlgorithmSnapshot,
) -> BoostedEnsembleReportMetrics:
    """Build boosted ensemble report metrics from a trainer snapshot."""
    return BoostedEnsembleReportMetrics(
        name=BOOSTED_ENSEMBLE_NAME,
        train_accuracy=float(snapshot.metrics["boosted_train_accuracy"]),
        test_accuracy=float(snapshot.metrics["boosted_test_accuracy"]),
        generalization_gap=float(snapshot.metrics["boosted_generalization_gap"]),
        mean_test_confidence=float(snapshot.metrics["mean_boosted_test_confidence"]),
        round_count=int(snapshot.metrics["completed_round_count"]),
        best_staged_test_accuracy=float(
            snapshot.metrics["best_staged_boosted_test_accuracy"],
        ),
        best_staged_round_index=int(snapshot.metrics["best_staged_round_index"]),
    )


def _winner_by_test_accuracy(
    *,
    weak_test_accuracy: float,
    boosted_test_accuracy: float,
) -> str:
    """Return model name with better test accuracy."""
    if boosted_test_accuracy >= weak_test_accuracy:
        return BOOSTED_ENSEMBLE_NAME

    return WEAK_LEARNER_NAME


def _build_summary(
    *,
    dataset_kind: str,
    winner: str,
    test_accuracy_delta: float,
    best_staged_round_index: int,
) -> str:
    """Build a compact report summary."""
    return (
        f"Summary: on {dataset_kind}, {winner} wins by test accuracy "
        f"with delta {test_accuracy_delta:+.3f}. "
        f"Best staged test accuracy was reached at round {best_staged_round_index}."
    )
