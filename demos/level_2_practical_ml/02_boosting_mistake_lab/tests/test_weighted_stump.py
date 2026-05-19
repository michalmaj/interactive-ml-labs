"""Tests showing that decision stumps use sample weights."""

import numpy as np
import pytest
from boosting_mistake_lab import (
    WeakLearnerBaseline,
    WeightedDatasetSplit,
    WeightedTrainTestDataset,
)
from ml_lab_core import DatasetSnapshot


def _weighted_tie_dataset() -> WeightedTrainTestDataset:
    """Create a tiny dataset where weights decide leaf predictions.

    The only valid useful split is around x1 = 0.

    Left side contains labels [0, 1].
    Right side contains labels [0, 1].

    Without weights, both sides are ties and class 0 would win.

    With weights:
    - left side should predict class 1,
    - right side should predict class 0.

    The second feature is constant on purpose, so the stump cannot find a
    perfect split along x2.
    """
    features = np.array(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    targets = np.array([0, 1, 0, 1], dtype=int)
    sample_weights = np.array([0.1, 0.4, 0.4, 0.1], dtype=float)

    split = WeightedDatasetSplit(
        snapshot=DatasetSnapshot(
            features=features,
            targets=targets,
            feature_names=("x1", "x2"),
            target_names=("class_0", "class_1"),
        ),
        sample_weights=sample_weights,
    )

    return WeightedTrainTestDataset(
        train=split,
        test=split,
        metadata={"dataset_kind": "weighted_tie"},
    )


def test_weighted_stump_uses_weighted_majority_classes() -> None:
    """Weighted leaf predictions should follow weighted majority labels."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_weighted_tie_dataset())
    split = learner.split

    assert split.feature_index == 0
    assert split.threshold == pytest.approx(0.0)
    assert split.left_prediction == 1
    assert split.right_prediction == 0
    assert split.weighted_training_error == pytest.approx(0.2)
    assert snapshot.metrics["weighted_train_error"] == pytest.approx(0.2)


def test_weighted_stump_split_metrics_are_exposed() -> None:
    """Snapshot should expose weighted split metrics."""
    learner = WeakLearnerBaseline()

    snapshot = learner.reset(_weighted_tie_dataset())

    assert "split_training_error" in snapshot.metrics
    assert "split_weighted_training_error" in snapshot.metrics
    assert snapshot.metrics["split_training_error"] == pytest.approx(0.5)
    assert snapshot.metrics["split_weighted_training_error"] == pytest.approx(0.2)


def test_weighted_stump_predictions_follow_weighted_split() -> None:
    """Predictions should use the fitted weighted stump split."""
    learner = WeakLearnerBaseline()
    learner.reset(_weighted_tie_dataset())

    predictions = learner.predict(
        np.array(
            [
                [-1.0, 0.5],
                [1.0, 0.5],
            ],
            dtype=float,
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([1, 0]))
