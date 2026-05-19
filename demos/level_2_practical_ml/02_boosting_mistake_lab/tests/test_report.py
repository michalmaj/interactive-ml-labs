"""Tests for Boosting Mistake Lab CLI reports."""

import pytest
from boosting_mistake_lab import (
    BOOSTED_ENSEMBLE_NAME,
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    WEAK_LEARNER_NAME,
    BoostingComparisonReport,
    BoostingTrainer,
    BoostingTrainerConfig,
    SyntheticWeightedDatasetConfig,
    WeakLearnerBaseline,
    build_boosting_comparison_report,
    format_boosting_comparison_report,
    make_synthetic_weighted_dataset,
)
from ml_lab_core import AlgorithmSnapshot

SAMPLES_PER_CLASS: int = 16
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_COUNT: int = 4


def _snapshots(dataset_kind: str):
    """Create dataset, weak snapshot, and trainer snapshot."""
    dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=dataset_kind,
        ),
    )
    weak_snapshot = WeakLearnerBaseline().reset(dataset)
    trainer_result = BoostingTrainer(
        BoostingTrainerConfig(round_count=ROUND_COUNT),
    ).reset(dataset)

    return dataset, weak_snapshot, trainer_result.snapshot


def test_build_boosting_comparison_report_returns_report() -> None:
    """Report builder should return a comparison report."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_XOR)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert isinstance(report, BoostingComparisonReport)


def test_build_boosting_comparison_report_contains_dataset_metadata() -> None:
    """Report should contain dataset metadata."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_AXIS_ALIGNED)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert report.dataset_kind == DATASET_KIND_AXIS_ALIGNED
    assert report.train_sample_count == SAMPLES_PER_CLASS * 2
    assert report.test_sample_count == SAMPLES_PER_CLASS * 2


def test_build_boosting_comparison_report_contains_weak_learner_metrics() -> None:
    """Report should contain weak learner metrics."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_XOR)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert report.weak_learner.name == WEAK_LEARNER_NAME
    assert report.weak_learner.train_accuracy == pytest.approx(
        weak_snapshot.metrics["train_accuracy"],
    )
    assert report.weak_learner.test_accuracy == pytest.approx(
        weak_snapshot.metrics["test_accuracy"],
    )
    assert report.weak_learner.weighted_train_error == pytest.approx(
        weak_snapshot.metrics["weighted_train_error"],
    )


def test_build_boosting_comparison_report_contains_boosted_metrics() -> None:
    """Report should contain boosted ensemble metrics."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_XOR)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert report.boosted_ensemble.name == BOOSTED_ENSEMBLE_NAME
    assert report.boosted_ensemble.round_count == ROUND_COUNT
    assert report.boosted_ensemble.train_accuracy == pytest.approx(
        trainer_snapshot.metrics["boosted_train_accuracy"],
    )
    assert report.boosted_ensemble.test_accuracy == pytest.approx(
        trainer_snapshot.metrics["boosted_test_accuracy"],
    )
    assert report.boosted_ensemble.best_staged_round_index == int(
        trainer_snapshot.metrics["best_staged_round_index"],
    )


def test_build_boosting_comparison_report_computes_deltas() -> None:
    """Report should compute boosted-minus-weak accuracy deltas."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_XOR)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert report.train_accuracy_delta == pytest.approx(
        report.boosted_ensemble.train_accuracy - report.weak_learner.train_accuracy,
    )
    assert report.test_accuracy_delta == pytest.approx(
        report.boosted_ensemble.test_accuracy - report.weak_learner.test_accuracy,
    )


def test_format_boosting_comparison_report_contains_key_sections() -> None:
    """Formatted report should contain readable report sections."""
    dataset, weak_snapshot, trainer_snapshot = _snapshots(DATASET_KIND_XOR)
    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    text = format_boosting_comparison_report(report)

    assert "Dataset:" in text
    assert "Weak learner baseline:" in text
    assert "Boosted ensemble:" in text
    assert "Comparison:" in text
    assert "winner by test acc" in text
    assert "Summary:" in text


def test_report_winner_is_boosted_when_boosted_test_accuracy_is_not_worse() -> None:
    """Boosted ensemble should win ties by test accuracy."""
    weak_snapshot = AlgorithmSnapshot(
        iteration=1,
        status="fitted",
        metrics={
            "train_accuracy": 0.80,
            "test_accuracy": 0.70,
            "weighted_train_error": 0.20,
            "learner_weight": 0.50,
            "feature_index": 0,
            "threshold": 0.0,
        },
        visual_state={},
        annotations=(),
        done=True,
    )
    trainer_snapshot = AlgorithmSnapshot(
        iteration=4,
        status="completed",
        metrics={
            "boosted_train_accuracy": 0.90,
            "boosted_test_accuracy": 0.70,
            "boosted_generalization_gap": 0.20,
            "mean_boosted_test_confidence": 0.75,
            "completed_round_count": 4,
            "best_staged_boosted_test_accuracy": 0.72,
            "best_staged_round_index": 3,
        },
        visual_state={},
        annotations=(),
        done=True,
    )
    dataset, _, _ = _snapshots(DATASET_KIND_XOR)

    report = build_boosting_comparison_report(
        dataset=dataset,
        weak_snapshot=weak_snapshot,
        trainer_snapshot=trainer_snapshot,
    )

    assert report.winner == BOOSTED_ENSEMBLE_NAME
