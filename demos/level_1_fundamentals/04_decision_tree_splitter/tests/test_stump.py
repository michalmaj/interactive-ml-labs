"""Tests for the decision stump model."""

import numpy as np
import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    DecisionStump,
    DecisionStumpConfig,
    LeafPrediction,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
PERFECT_ACCURACY: float = 1.0
XOR_STUMP_ACCURACY: float = 0.5
PERFECT_GINI_GAIN: float = 0.5


def _axis_dataset() -> DatasetSnapshot:
    """Create a low-noise axis-aligned dataset."""
    return make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(
            samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_AXIS_ALIGNED,
        ),
    )


def _xor_dataset() -> DatasetSnapshot:
    """Create a low-noise XOR dataset."""
    return make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(
            samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_XOR,
        ),
    )


def test_decision_stump_reset_returns_snapshot() -> None:
    """Reset should fit the stump and return a snapshot."""
    stump = DecisionStump()

    snapshot = stump.reset(_axis_dataset())

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "fitted"
    assert snapshot.done is True
    assert snapshot.iteration == 1


def test_decision_stump_fits_axis_aligned_dataset() -> None:
    """Decision stump should perfectly solve low-noise axis-aligned data."""
    stump = DecisionStump()

    snapshot = stump.reset(_axis_dataset())

    assert snapshot.metrics["feature_index"] == 0
    assert snapshot.metrics["training_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["information_gain"] == pytest.approx(PERFECT_GINI_GAIN)
    assert snapshot.metrics["left_prediction"] == 0
    assert snapshot.metrics["right_prediction"] == 1


def test_decision_stump_does_not_perfectly_solve_xor_at_root() -> None:
    """One stump should not solve XOR because one split is not enough."""
    stump = DecisionStump()

    snapshot = stump.reset(_xor_dataset())

    assert snapshot.metrics["training_accuracy"] == pytest.approx(XOR_STUMP_ACCURACY)
    assert float(snapshot.metrics["information_gain"]) < PERFECT_GINI_GAIN


def test_decision_stump_snapshot_contains_visual_state() -> None:
    """Snapshot should expose data needed by future renderers."""
    stump = DecisionStump()

    snapshot = stump.reset(_axis_dataset())

    assert "features" in snapshot.visual_state
    assert "targets" in snapshot.visual_state
    assert "predictions" in snapshot.visual_state
    assert "left_mask" in snapshot.visual_state
    assert "right_mask" in snapshot.visual_state
    assert "split_evaluation" in snapshot.visual_state
    assert "left_leaf" in snapshot.visual_state
    assert "right_leaf" in snapshot.visual_state


def test_decision_stump_exposes_leaf_predictions() -> None:
    """Fitted stump should expose leaf prediction objects."""
    stump = DecisionStump()
    stump.reset(_axis_dataset())

    assert isinstance(stump.left_leaf, LeafPrediction)
    assert isinstance(stump.right_leaf, LeafPrediction)
    assert stump.left_leaf.class_label == 0
    assert stump.right_leaf.class_label == 1
    assert stump.left_leaf.sample_count == SAMPLES_PER_CLASS
    assert stump.right_leaf.sample_count == SAMPLES_PER_CLASS


def test_decision_stump_predicts_labels() -> None:
    """Fitted stump should predict labels for query points."""
    stump = DecisionStump()
    stump.reset(_axis_dataset())

    predictions = stump.predict(
        np.array(
            [
                [-2.0, 0.0],
                [2.0, 0.0],
            ],
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_decision_stump_accepts_single_query_point() -> None:
    """Prediction should accept one query point."""
    stump = DecisionStump()
    stump.reset(_axis_dataset())

    prediction = stump.predict([2.0, 0.0])

    assert prediction.shape == (1,)
    assert prediction[0] == 1


def test_decision_stump_supports_entropy_criterion() -> None:
    """Decision stump should support entropy-based split selection."""
    stump = DecisionStump(DecisionStumpConfig(criterion="entropy"))

    snapshot = stump.reset(_axis_dataset())

    assert snapshot.metrics["criterion"] == "entropy"
    assert snapshot.metrics["training_accuracy"] == pytest.approx(PERFECT_ACCURACY)


def test_decision_stump_step_returns_snapshot() -> None:
    """Step should return the current fitted stump snapshot."""
    stump = DecisionStump()
    stump.reset(_axis_dataset())

    snapshot = stump.step()

    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_decision_stump_rejects_usage_before_reset() -> None:
    """Using stump before reset should fail clearly."""
    stump = DecisionStump()

    with pytest.raises(RuntimeError, match="reset"):
        stump.snapshot()


@pytest.mark.parametrize(
    "dataset, expected_message",
    [
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=None,
            ),
            "targets are required",
        ),
        (
            DatasetSnapshot(
                features=np.array([0.0, 1.0]),
                targets=np.array([0, 1]),
            ),
            "two-dimensional",
        ),
        (
            DatasetSnapshot(
                features=np.empty((0, 2)),
                targets=np.array([], dtype=int),
            ),
            "features cannot be empty",
        ),
        (
            DatasetSnapshot(
                features=np.empty((2, 0)),
                targets=np.array([0, 1]),
            ),
            "at least one column",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([[0], [1]]),
            ),
            "one-dimensional",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([0.0, 1.0]),
            ),
            "integers",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([0, -1]),
            ),
            "negative",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0], [1.0, 1.0]]),
                targets=np.array([0]),
            ),
            "same number of samples",
        ),
        (
            DatasetSnapshot(
                features=np.array([[0.0, 0.0]]),
                targets=np.array([0]),
            ),
            "At least two samples",
        ),
    ],
)
def test_decision_stump_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid datasets should fail clearly."""
    stump = DecisionStump()

    with pytest.raises(ValueError, match=expected_message):
        stump.reset(dataset)


@pytest.mark.parametrize(
    "features, expected_message",
    [
        (
            np.array([1.0, 2.0, 3.0]),
            "same number of columns",
        ),
        (
            np.empty((0, 2)),
            "features cannot be empty",
        ),
        (
            np.empty((1, 0)),
            "at least one column",
        ),
        (
            np.array([[1.0, 2.0, 3.0]]),
            "same number of columns",
        ),
    ],
)
def test_decision_stump_rejects_invalid_prediction_features(
    features: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid prediction features should fail clearly."""
    stump = DecisionStump()
    stump.reset(_axis_dataset())

    with pytest.raises(ValueError, match=expected_message):
        stump.predict(features)
