"""Tests for stepwise logistic regression."""

import numpy as np
import pytest
from logistic_regression_boundary_lab import (
    LogisticRegressionConfig,
    StepwiseLogisticRegression,
    SyntheticBinaryClassificationConfig,
    make_synthetic_binary_classification_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

TRAINING_STEPS: int = 30
LEARNING_RATE: float = 0.1
LOW_NOISE_STD: float = 0.3
SAMPLES_PER_CLASS: int = 20


def _dataset() -> DatasetSnapshot:
    """Create a reproducible low-noise dataset for model tests."""
    config = SyntheticBinaryClassificationConfig(
        samples_per_class=SAMPLES_PER_CLASS,
        noise_std=LOW_NOISE_STD,
        seed=123,
    )

    return make_synthetic_binary_classification_dataset(config)


def test_logistic_regression_can_reset_with_dataset() -> None:
    """Model should initialize its state from a dataset."""
    model = StepwiseLogisticRegression()

    model.reset(_dataset())
    snapshot = model.snapshot()

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.iteration == 0
    assert snapshot.metrics["loss"] > 0.0
    assert snapshot.metrics["threshold"] == pytest.approx(0.5)


def test_logistic_regression_step_reduces_loss() -> None:
    """Several gradient descent steps should reduce training loss."""
    model = StepwiseLogisticRegression(
        LogisticRegressionConfig(
            learning_rate=LEARNING_RATE,
            max_steps=TRAINING_STEPS,
        ),
    )
    model.reset(_dataset())

    initial_loss = float(model.snapshot().metrics["loss"])

    for _ in range(TRAINING_STEPS):
        model.step()

    final_loss = float(model.snapshot().metrics["loss"])

    assert final_loss < initial_loss


def test_logistic_regression_snapshot_contains_metrics_and_visual_state() -> None:
    """Snapshot should expose data needed by future renderers."""
    model = StepwiseLogisticRegression()
    model.reset(_dataset())

    snapshot = model.step()

    assert snapshot.status in {"running", "finished"}
    assert "features" in snapshot.visual_state
    assert "targets" in snapshot.visual_state
    assert "probabilities" in snapshot.visual_state
    assert "predictions" in snapshot.visual_state
    assert "weights" in snapshot.visual_state
    assert "loss_history" in snapshot.visual_state
    assert "loss" in snapshot.metrics
    assert "accuracy" in snapshot.metrics
    assert "precision" in snapshot.metrics
    assert "recall" in snapshot.metrics


def test_logistic_regression_predicts_probabilities() -> None:
    """Model should predict probabilities for query points."""
    model = StepwiseLogisticRegression()
    model.reset(_dataset())

    probabilities = model.predict_probabilities(
        np.array(
            [
                [-2.0, 0.0],
                [2.0, 0.0],
            ],
        ),
    )

    assert probabilities.shape == (2,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_logistic_regression_predicts_binary_labels() -> None:
    """Model should convert probabilities into binary labels."""
    model = StepwiseLogisticRegression(
        LogisticRegressionConfig(
            learning_rate=LEARNING_RATE,
            max_steps=TRAINING_STEPS,
        ),
    )
    model.reset(_dataset())

    for _ in range(TRAINING_STEPS):
        model.step()

    predictions = model.predict(
        np.array(
            [
                [-2.0, 0.0],
                [2.0, 0.0],
            ],
        ),
    )

    assert set(predictions.tolist()).issubset({0, 1})


def test_logistic_regression_accepts_single_query_point() -> None:
    """Prediction helpers should accept one query point."""
    model = StepwiseLogisticRegression()
    model.reset(_dataset())

    probabilities = model.predict_probabilities([0.0, 0.0])
    predictions = model.predict([0.0, 0.0])

    assert probabilities.shape == (1,)
    assert predictions.shape == (1,)


def test_logistic_regression_rejects_prediction_before_reset() -> None:
    """Prediction before reset should fail clearly."""
    model = StepwiseLogisticRegression()

    with pytest.raises(RuntimeError, match="reset"):
        model.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            LogisticRegressionConfig(learning_rate=0.0),
            "learning_rate must be greater than 0",
        ),
        (
            LogisticRegressionConfig(threshold=-0.1),
            "threshold must be in the range",
        ),
        (
            LogisticRegressionConfig(threshold=1.1),
            "threshold must be in the range",
        ),
        (
            LogisticRegressionConfig(max_steps=0),
            "max_steps must be greater than 0",
        ),
        (
            LogisticRegressionConfig(convergence_tolerance=-1.0),
            "convergence_tolerance cannot be negative",
        ),
    ],
)
def test_logistic_regression_rejects_invalid_config(
    config: LogisticRegressionConfig,
    expected_message: str,
) -> None:
    """Invalid model configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        StepwiseLogisticRegression(config)


@pytest.mark.parametrize(
    "dataset, expected_message",
    [
        (
            DatasetSnapshot(features=np.array([1.0, 2.0]), targets=np.array([0, 1])),
            "two-dimensional",
        ),
        (
            DatasetSnapshot(features=np.empty((0, 2)), targets=np.array([])),
            "features cannot be empty",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0, 3.0]]), targets=np.array([0])),
            "exactly two columns",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0]]), targets=None),
            "targets are required",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0]]), targets=np.array([[0]])),
            "one-dimensional",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0]]), targets=np.array([2])),
            "binary labels",
        ),
        (
            DatasetSnapshot(
                features=np.array([[1.0, 2.0], [3.0, 4.0]]),
                targets=np.array([0]),
            ),
            "same number of samples",
        ),
    ],
)
def test_logistic_regression_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid datasets should fail clearly."""
    model = StepwiseLogisticRegression()

    with pytest.raises(ValueError, match=expected_message):
        model.reset(dataset)


@pytest.mark.parametrize(
    "features, expected_message",
    [
        (
            np.array([1.0, 2.0, 3.0]),
            "exactly two columns",
        ),
        (
            np.empty((0, 2)),
            "features cannot be empty",
        ),
        (
            np.array([[1.0, 2.0, 3.0]]),
            "exactly two columns",
        ),
    ],
)
def test_logistic_regression_rejects_invalid_prediction_features(
    features: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid prediction features should fail clearly."""
    model = StepwiseLogisticRegression()
    model.reset(_dataset())

    with pytest.raises(ValueError, match=expected_message):
        model.predict_probabilities(features)
