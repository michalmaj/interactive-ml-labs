"""Tests for stepwise linear regression."""

import numpy as np
import pytest
from gradient_descent_playground import (
    GradientDescentConfig,
    StepwiseLinearRegression,
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot, StepwiseAlgorithm

SAMPLE_COUNT: int = 120
TRUE_WEIGHT: float = 2.0
TRUE_BIAS: float = -1.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 7
LEARNING_RATE: float = 0.03
MAX_STEPS: int = 50
SHORT_MAX_STEPS: int = 2


def test_stepwise_linear_regression_matches_protocol() -> None:
    """StepwiseLinearRegression should satisfy the StepwiseAlgorithm protocol."""
    model = StepwiseLinearRegression()

    assert isinstance(model, StepwiseAlgorithm)


def test_stepwise_linear_regression_initial_snapshot_contains_loss() -> None:
    """After reset, the model should expose an initial snapshot."""
    dataset = make_synthetic_regression_dataset(
        SyntheticRegressionConfig(
            sample_count=SAMPLE_COUNT,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
        ),
    )
    model = StepwiseLinearRegression()

    model.reset(dataset)
    snapshot = model.snapshot()

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.iteration == 0
    assert snapshot.metrics["loss"] > 0.0
    assert snapshot.metrics["weight"] == 0.0
    assert snapshot.metrics["bias"] == 0.0
    assert snapshot.done is False


def test_stepwise_linear_regression_reduces_loss_after_multiple_steps() -> None:
    """Gradient descent should reduce MSE on a simple linear dataset."""
    dataset = make_synthetic_regression_dataset(
        SyntheticRegressionConfig(
            sample_count=SAMPLE_COUNT,
            true_weight=TRUE_WEIGHT,
            true_bias=TRUE_BIAS,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
        ),
    )
    model = StepwiseLinearRegression(
        GradientDescentConfig(
            learning_rate=LEARNING_RATE,
            max_steps=MAX_STEPS,
        ),
    )

    model.reset(dataset)
    initial_loss = float(model.snapshot().metrics["loss"])

    for _ in range(MAX_STEPS):
        snapshot = model.step()

    final_loss = float(snapshot.metrics["loss"])

    assert final_loss < initial_loss
    assert model.weight > 0.0
    assert model.bias < 0.0


def test_stepwise_linear_regression_stops_after_max_steps() -> None:
    """The model should finish when max_steps is reached."""
    dataset = make_synthetic_regression_dataset(
        SyntheticRegressionConfig(sample_count=SAMPLE_COUNT, seed=SEED),
    )
    model = StepwiseLinearRegression(
        GradientDescentConfig(
            learning_rate=LEARNING_RATE,
            max_steps=SHORT_MAX_STEPS,
        ),
    )

    model.reset(dataset)
    first = model.step()
    second = model.step()

    assert first.done is False
    assert second.done is True
    assert second.iteration == SHORT_MAX_STEPS


def test_stepwise_linear_regression_predicts_after_reset() -> None:
    """The model should predict values using current weight and bias."""
    dataset = make_synthetic_regression_dataset(
        SyntheticRegressionConfig(sample_count=SAMPLE_COUNT, seed=SEED),
    )
    model = StepwiseLinearRegression(
        GradientDescentConfig(
            initial_weight=TRUE_WEIGHT,
            initial_bias=TRUE_BIAS,
        ),
    )

    model.reset(dataset)

    predictions = model.predict(np.array([[1.0], [2.0], [3.0]]))

    np.testing.assert_allclose(predictions, np.array([1.0, 3.0, 5.0]))


def test_stepwise_linear_regression_rejects_step_before_reset() -> None:
    """Calling step before reset should fail clearly."""
    model = StepwiseLinearRegression()

    with pytest.raises(RuntimeError, match="reset"):
        model.step()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            GradientDescentConfig(learning_rate=0.0),
            "learning_rate must be greater than 0",
        ),
        (
            GradientDescentConfig(max_steps=0),
            "max_steps must be greater than 0",
        ),
        (
            GradientDescentConfig(convergence_tolerance=-1.0),
            "convergence_tolerance cannot be negative",
        ),
    ],
)
def test_stepwise_linear_regression_rejects_invalid_config(
    config: GradientDescentConfig,
    expected_message: str,
) -> None:
    """Invalid gradient descent configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        StepwiseLinearRegression(config)


@pytest.mark.parametrize(
    "dataset, expected_message",
    [
        (
            DatasetSnapshot(features=np.array([1.0, 2.0, 3.0]), targets=np.array([1.0, 2.0, 3.0])),
            "two-dimensional array",
        ),
        (
            DatasetSnapshot(
                features=np.array([[1.0, 2.0], [3.0, 4.0]]),
                targets=np.array([1.0, 2.0]),
            ),
            "exactly one column",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0], [2.0]]), targets=None),
            "targets are required",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0], [2.0]]), targets=np.array([[1.0], [2.0]])),
            "one-dimensional array",
        ),
        (
            DatasetSnapshot(features=np.empty((0, 1)), targets=np.array([])),
            "cannot be empty",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0], [2.0]]), targets=np.array([1.0])),
            "same number of samples",
        ),
    ],
)
def test_stepwise_linear_regression_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid datasets should fail clearly during reset."""
    model = StepwiseLinearRegression()

    with pytest.raises(ValueError, match=expected_message):
        model.reset(dataset)
