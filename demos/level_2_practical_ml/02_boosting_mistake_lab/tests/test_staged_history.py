"""Tests for staged boosted accuracy history."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_XOR,
    BoostingTrainer,
    BoostingTrainerConfig,
    StagedBoostingHistory,
    SyntheticWeightedDatasetConfig,
    build_staged_boosting_history,
    make_synthetic_weighted_dataset,
)

SAMPLES_PER_CLASS: int = 20
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_COUNT: int = 4


def _trainer_result():
    """Create a deterministic trainer result."""
    dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_XOR,
        ),
    )
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))
    result = trainer.reset(dataset)

    return dataset, result


def test_build_staged_boosting_history_returns_history() -> None:
    """Staged history builder should return a history object."""
    dataset, result = _trainer_result()

    history = build_staged_boosting_history(
        round_results=result.round_results,
        train_features=dataset.train.snapshot.features,
        train_targets=dataset.train.snapshot.targets,
        test_features=dataset.test.snapshot.features,
        test_targets=dataset.test.snapshot.targets,
    )

    assert isinstance(history, StagedBoostingHistory)


def test_staged_boosting_history_has_expected_shapes() -> None:
    """Staged history arrays should have one value per round."""
    _, result = _trainer_result()
    history = result.staged_history

    assert history.round_indices.shape == (ROUND_COUNT,)
    assert history.boosted_train_accuracies.shape == (ROUND_COUNT,)
    assert history.boosted_test_accuracies.shape == (ROUND_COUNT,)
    assert history.boosted_generalization_gaps.shape == (ROUND_COUNT,)
    assert history.mean_train_confidences.shape == (ROUND_COUNT,)
    assert history.mean_test_confidences.shape == (ROUND_COUNT,)
    assert history.learner_weights.shape == (ROUND_COUNT,)
    assert history.weighted_train_errors.shape == (ROUND_COUNT,)


def test_staged_boosting_history_round_indices_are_one_based() -> None:
    """Round indices should match completed boosting rounds."""
    _, result = _trainer_result()

    np.testing.assert_array_equal(
        result.staged_history.round_indices,
        np.array([1, 2, 3, 4]),
    )


def test_staged_boosting_history_values_are_bounded() -> None:
    """Staged accuracy and confidence values should be in valid ranges."""
    _, result = _trainer_result()
    history = result.staged_history

    assert np.all(history.boosted_train_accuracies >= 0.0)
    assert np.all(history.boosted_train_accuracies <= 1.0)
    assert np.all(history.boosted_test_accuracies >= 0.0)
    assert np.all(history.boosted_test_accuracies <= 1.0)
    assert np.all(history.mean_train_confidences >= 0.5)
    assert np.all(history.mean_train_confidences <= 1.0)
    assert np.all(history.mean_test_confidences >= 0.5)
    assert np.all(history.mean_test_confidences <= 1.0)


def test_staged_boosting_history_final_properties_match_last_values() -> None:
    """Final properties should expose the last staged values."""
    _, result = _trainer_result()
    history = result.staged_history

    assert history.final_train_accuracy == pytest.approx(
        history.boosted_train_accuracies[-1],
    )
    assert history.final_test_accuracy == pytest.approx(
        history.boosted_test_accuracies[-1],
    )
    assert history.final_generalization_gap == pytest.approx(
        history.boosted_generalization_gaps[-1],
    )


def test_build_staged_boosting_history_rejects_empty_rounds() -> None:
    """Staged history requires at least one round."""
    dataset, _ = _trainer_result()

    with pytest.raises(ValueError, match="round_results"):
        build_staged_boosting_history(
            round_results=[],
            train_features=dataset.train.snapshot.features,
            train_targets=dataset.train.snapshot.targets,
            test_features=dataset.test.snapshot.features,
            test_targets=dataset.test.snapshot.targets,
        )


def test_build_staged_boosting_history_rejects_feature_target_mismatch() -> None:
    """Feature and target sample counts must match."""
    dataset, result = _trainer_result()

    with pytest.raises(ValueError, match="same number of samples"):
        build_staged_boosting_history(
            round_results=result.round_results,
            train_features=dataset.train.snapshot.features[:2],
            train_targets=dataset.train.snapshot.targets,
            test_features=dataset.test.snapshot.features,
            test_targets=dataset.test.snapshot.targets,
        )
