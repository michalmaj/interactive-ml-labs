"""Tests for one complete boosting round."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    BoostingRoundConfig,
    BoostingRoundResult,
    SyntheticWeightedDatasetConfig,
    WeightedTrainTestDataset,
    make_next_round_dataset,
    make_synthetic_weighted_dataset,
    run_boosting_round,
)
from ml_lab_core import AlgorithmSnapshot

SAMPLES_PER_CLASS: int = 20
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_INDEX: int = 3


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


def test_run_boosting_round_returns_result() -> None:
    """Running one boosting round should return a round result."""
    result = run_boosting_round(_dataset(DATASET_KIND_AXIS_ALIGNED))

    assert isinstance(result, BoostingRoundResult)
    assert isinstance(result.weak_snapshot, AlgorithmSnapshot)
    assert isinstance(result.round_snapshot, AlgorithmSnapshot)
    assert isinstance(result.next_dataset, WeightedTrainTestDataset)


def test_run_boosting_round_uses_configured_round_index() -> None:
    """Round snapshot should use configured round index."""
    result = run_boosting_round(
        _dataset(DATASET_KIND_XOR),
        BoostingRoundConfig(round_index=ROUND_INDEX),
    )

    assert result.round_index == ROUND_INDEX
    assert result.round_snapshot.iteration == ROUND_INDEX
    assert result.round_snapshot.metrics["round_index"] == ROUND_INDEX
    assert result.next_dataset.metadata["last_completed_boosting_round"] == ROUND_INDEX


def test_run_boosting_round_exposes_core_metrics() -> None:
    """Round snapshot should expose boosting round metrics."""
    result = run_boosting_round(_dataset(DATASET_KIND_XOR))
    metrics = result.round_snapshot.metrics

    assert "weighted_train_error" in metrics
    assert "weighted_train_accuracy" in metrics
    assert "learner_weight" in metrics
    assert "old_mistake_weight_sum" in metrics
    assert "updated_mistake_weight_sum" in metrics
    assert "weight_l1_change" in metrics
    assert 0.0 <= float(metrics["weighted_train_error"]) <= 1.0
    assert float(metrics["weight_l1_change"]) >= 0.0


def test_run_boosting_round_exposes_visual_state() -> None:
    """Round snapshot should expose visual state for future UI."""
    result = run_boosting_round(_dataset(DATASET_KIND_XOR))
    visual_state = result.round_snapshot.visual_state

    assert "weak_snapshot" in visual_state
    assert "train_features" in visual_state
    assert "train_targets" in visual_state
    assert "train_predictions" in visual_state
    assert "train_mistakes" in visual_state
    assert "current_train_sample_weights" in visual_state
    assert "updated_train_sample_weights" in visual_state
    assert "test_features" in visual_state
    assert "test_targets" in visual_state
    assert "test_predictions" in visual_state
    assert "test_mistakes" in visual_state
    assert "split" in visual_state
    assert "learner_weight_result" in visual_state
    assert "weight_update_result" in visual_state


def test_run_boosting_round_updates_next_dataset_train_weights() -> None:
    """Next-round dataset should contain updated train weights."""
    dataset = _dataset(DATASET_KIND_XOR)

    result = run_boosting_round(dataset)

    current_weights = np.asarray(dataset.train.sample_weights)
    updated_weights = np.asarray(result.next_dataset.train.sample_weights)

    assert np.sum(updated_weights) == pytest.approx(1.0)
    np.testing.assert_allclose(updated_weights, result.updated_train_weights)
    assert updated_weights.shape == current_weights.shape


def test_run_boosting_round_preserves_test_split() -> None:
    """Next-round dataset should keep test split unchanged."""
    dataset = _dataset(DATASET_KIND_XOR)

    result = run_boosting_round(dataset)

    np.testing.assert_allclose(
        result.next_dataset.test.sample_weights,
        dataset.test.sample_weights,
    )
    np.testing.assert_allclose(
        result.next_dataset.test.snapshot.features,
        dataset.test.snapshot.features,
    )
    np.testing.assert_array_equal(
        result.next_dataset.test.snapshot.targets,
        dataset.test.snapshot.targets,
    )


def test_run_boosting_round_result_properties_match_snapshot_metrics() -> None:
    """Convenience properties should read values from round snapshot."""
    result = run_boosting_round(_dataset(DATASET_KIND_XOR))

    assert result.learner_weight == pytest.approx(
        result.round_snapshot.metrics["learner_weight"],
    )
    assert result.weighted_train_error == pytest.approx(
        result.round_snapshot.metrics["weighted_train_error"],
    )


def test_make_next_round_dataset_rejects_invalid_weight_shape() -> None:
    """Next-round dataset requires weights matching train sample count."""
    dataset = _dataset(DATASET_KIND_XOR)

    with pytest.raises(ValueError, match="must match sample count"):
        make_next_round_dataset(
            dataset=dataset,
            updated_train_sample_weights=np.array([1.0]),
            round_index=1,
        )


def test_make_next_round_dataset_rejects_unnormalized_weights() -> None:
    """Next-round dataset requires normalized weights."""
    dataset = _dataset(DATASET_KIND_XOR)
    sample_count = np.asarray(dataset.train.snapshot.targets).shape[0]

    with pytest.raises(ValueError, match=r"sum to 1.0"):
        make_next_round_dataset(
            dataset=dataset,
            updated_train_sample_weights=np.ones(sample_count),
            round_index=1,
        )


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            BoostingRoundConfig(round_index=0),
            "round_index",
        ),
        (
            BoostingRoundConfig(min_samples_leaf=0),
            "min_samples_leaf",
        ),
    ],
)
def test_run_boosting_round_rejects_invalid_config(
    config: BoostingRoundConfig,
    expected_message: str,
) -> None:
    """Invalid boosting round config should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        run_boosting_round(_dataset(DATASET_KIND_XOR), config)
