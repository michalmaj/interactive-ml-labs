"""Tests for boosted ensemble prediction."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    BoostedPredictionResult,
    BoostingTrainer,
    BoostingTrainerConfig,
    SyntheticWeightedDatasetConfig,
    boosted_accuracy_score,
    make_synthetic_weighted_dataset,
    predict_boosted_ensemble,
)

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_COUNT: int = 3


def _trainer_result() -> tuple[BoostingTrainer, np.ndarray]:
    """Fit a small trainer and return trainer with train features."""
    dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_AXIS_ALIGNED,
        ),
    )
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))
    trainer.reset(dataset)

    return trainer, np.asarray(dataset.train.snapshot.features, dtype=float)


def test_predict_boosted_ensemble_returns_result() -> None:
    """Boosted prediction should return a result object."""
    trainer, features = _trainer_result()
    result = trainer.result()

    prediction = predict_boosted_ensemble(
        weak_learners=[round_result.weak_learner for round_result in result.round_results],
        learner_weights=result.learner_weights,
        features=features,
    )

    assert isinstance(prediction, BoostedPredictionResult)


def test_predict_boosted_ensemble_returns_expected_shapes() -> None:
    """Boosted prediction should return arrays with expected shapes."""
    trainer, features = _trainer_result()
    result = trainer.result()

    prediction = predict_boosted_ensemble(
        weak_learners=[round_result.weak_learner for round_result in result.round_results],
        learner_weights=result.learner_weights,
        features=features,
    )

    assert prediction.predictions.shape == (features.shape[0],)
    assert prediction.confidence.shape == (features.shape[0],)
    assert prediction.raw_scores.shape == (features.shape[0],)
    assert prediction.normalized_margins.shape == (features.shape[0],)
    assert prediction.learner_predictions.shape == (ROUND_COUNT, features.shape[0])
    assert prediction.learner_weights.shape == (ROUND_COUNT,)


def test_predict_boosted_ensemble_confidence_is_bounded() -> None:
    """Boosted confidence should be in the range [0.5, 1.0]."""
    trainer, features = _trainer_result()
    result = trainer.result()

    prediction = predict_boosted_ensemble(
        weak_learners=[round_result.weak_learner for round_result in result.round_results],
        learner_weights=result.learner_weights,
        features=features,
    )

    assert np.all(prediction.confidence >= 0.5)
    assert np.all(prediction.confidence <= 1.0)


def test_predict_boosted_ensemble_accepts_single_feature_vector() -> None:
    """Boosted prediction should accept one feature vector."""
    trainer, features = _trainer_result()
    result = trainer.result()

    prediction = predict_boosted_ensemble(
        weak_learners=[round_result.weak_learner for round_result in result.round_results],
        learner_weights=result.learner_weights,
        features=features[0],
    )

    assert prediction.predictions.shape == (1,)


def test_boosted_accuracy_score_computes_accuracy() -> None:
    """Boosted accuracy helper should compute classification accuracy."""
    accuracy = boosted_accuracy_score(
        y_true=np.array([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
    )

    assert accuracy == pytest.approx(0.75)


@pytest.mark.parametrize(
    "learner_weights, expected_message",
    [
        (
            np.array([[1.0, 2.0]]),
            "learner_weights must be a one-dimensional array",
        ),
        (
            np.array([1.0]),
            "learner_weights must match weak learner count",
        ),
        (
            np.array([1.0, np.inf, 2.0]),
            "learner_weights must contain finite values",
        ),
    ],
)
def test_predict_boosted_ensemble_rejects_invalid_learner_weights(
    learner_weights: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid learner weights should fail clearly."""
    trainer, features = _trainer_result()
    result = trainer.result()

    with pytest.raises(ValueError, match=expected_message):
        predict_boosted_ensemble(
            weak_learners=[round_result.weak_learner for round_result in result.round_results],
            learner_weights=learner_weights,
            features=features,
        )


def test_predict_boosted_ensemble_rejects_empty_weak_learner_sequence() -> None:
    """Boosted prediction requires at least one weak learner."""
    with pytest.raises(ValueError, match="weak_learners"):
        predict_boosted_ensemble(
            weak_learners=[],
            learner_weights=np.array([], dtype=float),
            features=np.array([[0.0, 0.0]], dtype=float),
        )


@pytest.mark.parametrize(
    "y_true, y_pred, expected_message",
    [
        (
            np.array([[0, 1]]),
            np.array([0, 1]),
            "y_true must be a one-dimensional array",
        ),
        (
            np.array([0, 1]),
            np.array([[0, 1]]),
            "y_pred must be a one-dimensional array",
        ),
        (
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            "y_true must contain integers",
        ),
        (
            np.array([0, 2]),
            np.array([0, 1]),
            "y_true must contain binary labels",
        ),
        (
            np.array([0, 1]),
            np.array([0]),
            "same number of samples",
        ),
    ],
)
def test_boosted_accuracy_score_rejects_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid accuracy inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        boosted_accuracy_score(y_true=y_true, y_pred=y_pred)
