"""Tests for logistic regression confusion matrix snapshot metrics."""

from logistic_regression_boundary_lab import (
    LogisticRegressionConfig,
    StepwiseLogisticRegression,
    SyntheticBinaryClassificationConfig,
    make_synthetic_binary_classification_dataset,
)

SAMPLES_PER_CLASS: int = 20
LOW_NOISE_STD: float = 0.3
TRAINING_STEPS: int = 10


def test_logistic_regression_snapshot_contains_confusion_matrix_counts() -> None:
    """Snapshot metrics should expose binary confusion matrix counts."""
    dataset = make_synthetic_binary_classification_dataset(
        SyntheticBinaryClassificationConfig(
            samples_per_class=SAMPLES_PER_CLASS,
            noise_std=LOW_NOISE_STD,
            seed=123,
        ),
    )
    model = StepwiseLogisticRegression(
        LogisticRegressionConfig(
            learning_rate=0.1,
            max_steps=TRAINING_STEPS,
            threshold=0.5,
        ),
    )

    model.reset(dataset)

    for _ in range(TRAINING_STEPS):
        model.step()

    snapshot = model.snapshot()

    true_positive = int(snapshot.metrics["true_positive"])
    true_negative = int(snapshot.metrics["true_negative"])
    false_positive = int(snapshot.metrics["false_positive"])
    false_negative = int(snapshot.metrics["false_negative"])

    assert true_positive >= 0
    assert true_negative >= 0
    assert false_positive >= 0
    assert false_negative >= 0

    assert true_positive + true_negative + false_positive + false_negative == SAMPLES_PER_CLASS * 2
