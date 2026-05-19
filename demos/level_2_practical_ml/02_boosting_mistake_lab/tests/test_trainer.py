"""Tests for the multi-round boosting trainer."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    BoostingRoundResult,
    BoostingTrainer,
    BoostingTrainerConfig,
    BoostingTrainerResult,
    SyntheticWeightedDatasetConfig,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
)
from ml_lab_core import AlgorithmSnapshot

SAMPLES_PER_CLASS: int = 20
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_COUNT: int = 4


def _dataset(dataset_kind: str) -> WeightedTrainTestDataset:
    """Create a deterministic weighted train/test dataset."""
    return make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=dataset_kind,
        ),
    )


def test_boosting_trainer_returns_result() -> None:
    """Trainer reset should return a trainer result."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert isinstance(result, BoostingTrainerResult)
    assert isinstance(result.snapshot, AlgorithmSnapshot)
    assert isinstance(result.final_dataset, WeightedTrainTestDataset)


def test_boosting_trainer_runs_configured_number_of_rounds() -> None:
    """Trainer should run the configured number of rounds."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert result.round_count == ROUND_COUNT
    assert len(result.round_results) == ROUND_COUNT
    assert result.snapshot.metrics["completed_round_count"] == ROUND_COUNT


def test_boosting_trainer_round_indices_are_sequential() -> None:
    """Round results should have one-based sequential indices."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert [round_result.round_index for round_result in result.round_results] == [1, 2, 3, 4]
    assert all(
        isinstance(round_result, BoostingRoundResult) for round_result in result.round_results
    )


def test_boosting_trainer_passes_updated_weights_between_rounds() -> None:
    """Each round should start with weights produced by the previous round."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    for previous_round, next_round in zip(
        result.round_results[:-1],
        result.round_results[1:],
        strict=True,
    ):
        previous_updated_weights = np.asarray(
            previous_round.round_snapshot.visual_state["updated_train_sample_weights"],
            dtype=float,
        )
        next_current_weights = np.asarray(
            next_round.round_snapshot.visual_state["current_train_sample_weights"],
            dtype=float,
        )

        np.testing.assert_allclose(next_current_weights, previous_updated_weights)


def test_boosting_trainer_final_weights_are_normalized() -> None:
    """Final train weights should remain normalized."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert np.sum(result.final_train_weights) == pytest.approx(1.0)
    assert np.sum(result.final_dataset.train.sample_weights) == pytest.approx(1.0)


def test_boosting_trainer_result_properties_expose_stage_arrays() -> None:
    """Result should expose learner weights and weighted errors."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert result.learner_weights.shape == (ROUND_COUNT,)
    assert result.weighted_train_errors.shape == (ROUND_COUNT,)
    assert result.final_train_weights.shape == result.initial_dataset.train.sample_weights.shape


def test_boosting_trainer_snapshot_contains_metrics() -> None:
    """Trainer snapshot should expose aggregate metrics."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    metrics = result.snapshot.metrics

    assert metrics["round_count"] == ROUND_COUNT
    assert metrics["final_round_index"] == ROUND_COUNT
    assert "final_weighted_train_error" in metrics
    assert "final_learner_weight" in metrics
    assert "mean_weighted_train_error" in metrics
    assert "mean_learner_weight" in metrics
    assert "cumulative_weight_l1_change" in metrics
    assert "min_final_train_weight" in metrics
    assert "max_final_train_weight" in metrics


def test_boosting_trainer_snapshot_contains_visual_state() -> None:
    """Trainer snapshot should expose staged arrays for future visualization."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    visual_state = result.snapshot.visual_state

    assert "round_snapshots" in visual_state
    assert "weak_snapshots" in visual_state
    assert "staged_learner_weights" in visual_state
    assert "staged_weighted_train_errors" in visual_state
    assert "staged_weight_l1_changes" in visual_state
    assert "staged_train_accuracies" in visual_state
    assert "staged_test_accuracies" in visual_state
    assert "initial_train_sample_weights" in visual_state
    assert "final_train_sample_weights" in visual_state
    assert "final_dataset" in visual_state

    assert len(visual_state["round_snapshots"]) == ROUND_COUNT
    assert np.asarray(visual_state["staged_learner_weights"]).shape == (ROUND_COUNT,)


def test_boosting_trainer_stores_latest_result() -> None:
    """Trainer should expose the latest result and snapshot after reset."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert trainer.result() is result
    assert trainer.snapshot() is result.snapshot


def test_boosting_trainer_rejects_usage_before_reset() -> None:
    """Using trainer before fitting should fail clearly."""
    trainer = BoostingTrainer()

    with pytest.raises(RuntimeError, match="reset"):
        trainer.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            BoostingTrainerConfig(round_count=0),
            "round_count",
        ),
        (
            BoostingTrainerConfig(min_samples_leaf=0),
            "min_samples_leaf",
        ),
    ],
)
def test_boosting_trainer_rejects_invalid_config(
    config: BoostingTrainerConfig,
    expected_message: str,
) -> None:
    """Invalid trainer config should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        BoostingTrainer(config)


def test_boosting_trainer_snapshot_contains_boosted_prediction_metrics() -> None:
    """Trainer snapshot should expose final boosted prediction metrics."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    metrics = result.snapshot.metrics

    assert "boosted_train_accuracy" in metrics
    assert "boosted_test_accuracy" in metrics
    assert "boosted_generalization_gap" in metrics
    assert "mean_boosted_train_confidence" in metrics
    assert "mean_boosted_test_confidence" in metrics
    assert 0.0 <= float(metrics["boosted_train_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["boosted_test_accuracy"]) <= 1.0


def test_boosting_trainer_snapshot_contains_boosted_prediction_visual_state() -> None:
    """Trainer snapshot should expose boosted predictions for future UI."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    visual_state = result.snapshot.visual_state

    assert "boosted_train_predictions" in visual_state
    assert "boosted_test_predictions" in visual_state
    assert "boosted_train_confidence" in visual_state
    assert "boosted_test_confidence" in visual_state
    assert "boosted_train_scores" in visual_state
    assert "boosted_test_scores" in visual_state
    assert "boosted_train_result" in visual_state
    assert "boosted_test_result" in visual_state


def test_boosting_trainer_result_exposes_staged_boosted_accuracies() -> None:
    """Trainer result should expose staged boosted accuracy arrays."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))

    assert result.boosted_train_accuracies.shape == (ROUND_COUNT,)
    assert result.boosted_test_accuracies.shape == (ROUND_COUNT,)


def test_boosting_trainer_snapshot_contains_staged_boosted_history() -> None:
    """Trainer snapshot should expose staged boosted history for future plots."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    visual_state = result.snapshot.visual_state

    assert "staged_history" in visual_state
    assert "staged_boosted_train_accuracies" in visual_state
    assert "staged_boosted_test_accuracies" in visual_state
    assert "staged_boosted_generalization_gaps" in visual_state
    assert "staged_mean_train_confidences" in visual_state
    assert "staged_mean_test_confidences" in visual_state


def test_boosting_trainer_snapshot_contains_best_staged_metrics() -> None:
    """Trainer snapshot should expose best staged boosted test accuracy."""
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))

    result = trainer.reset(_dataset(DATASET_KIND_XOR))
    metrics = result.snapshot.metrics

    assert "best_staged_boosted_test_accuracy" in metrics
    assert "best_staged_round_index" in metrics
    assert 1 <= int(metrics["best_staged_round_index"]) <= ROUND_COUNT
