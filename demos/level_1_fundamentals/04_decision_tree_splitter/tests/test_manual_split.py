"""Tests for the manual split prototype."""

import numpy as np
import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    ManualSplitConfig,
    ManualSplitPrototype,
    SplitCandidate,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
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


def test_manual_split_reset_returns_snapshot() -> None:
    """Reset should evaluate the configured split and return a snapshot."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )

    snapshot = prototype.reset(_axis_dataset())

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "ready"
    assert snapshot.done is True
    assert snapshot.iteration == 0


def test_manual_split_snapshot_contains_metrics_and_visual_state() -> None:
    """Snapshot should expose data needed by future renderers."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )

    snapshot = prototype.reset(_axis_dataset())

    assert "features" in snapshot.visual_state
    assert "targets" in snapshot.visual_state
    assert "left_mask" in snapshot.visual_state
    assert "right_mask" in snapshot.visual_state
    assert "candidate" in snapshot.visual_state
    assert "criterion" in snapshot.visual_state

    assert snapshot.metrics["feature_index"] == 0
    assert snapshot.metrics["threshold"] == pytest.approx(0.0)
    assert snapshot.metrics["criterion"] == "gini"
    assert "parent_impurity" in snapshot.metrics
    assert "left_impurity" in snapshot.metrics
    assert "right_impurity" in snapshot.metrics
    assert "weighted_child_impurity" in snapshot.metrics
    assert "information_gain" in snapshot.metrics
    assert "left_sample_count" in snapshot.metrics
    assert "right_sample_count" in snapshot.metrics


def test_manual_split_can_evaluate_perfect_axis_aligned_split() -> None:
    """Manual split x1 <= 0 should solve low-noise axis-aligned data."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )

    snapshot = prototype.reset(_axis_dataset())

    assert snapshot.metrics["parent_impurity"] == pytest.approx(0.5)
    assert snapshot.metrics["left_impurity"] == pytest.approx(0.0)
    assert snapshot.metrics["right_impurity"] == pytest.approx(0.0)
    assert snapshot.metrics["information_gain"] == pytest.approx(PERFECT_GINI_GAIN)


def test_manual_split_on_xor_is_not_perfect_at_root() -> None:
    """One manual split should not perfectly solve XOR at the root."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )

    snapshot = prototype.reset(_xor_dataset())

    assert float(snapshot.metrics["information_gain"]) < PERFECT_GINI_GAIN


def test_manual_split_can_update_candidate() -> None:
    """Prototype should evaluate a new valid split candidate."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )
    prototype.reset(_axis_dataset())

    snapshot = prototype.set_split(
        SplitCandidate(
            feature_index=0,
            threshold=-1.0,
        ),
    )

    assert snapshot.iteration == 1
    assert snapshot.metrics["feature_index"] == 0
    assert snapshot.metrics["threshold"] == pytest.approx(-1.0)


def test_manual_split_can_update_criterion() -> None:
    """Prototype should support changing impurity criterion."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )
    prototype.reset(_axis_dataset())

    snapshot = prototype.set_split(
        SplitCandidate(
            feature_index=0,
            threshold=0.0,
        ),
        criterion="entropy",
    )

    assert snapshot.metrics["criterion"] == "entropy"
    assert snapshot.metrics["parent_impurity"] == pytest.approx(1.0)


def test_manual_split_rejects_usage_before_reset() -> None:
    """Prototype should fail clearly before dataset reset."""
    prototype = ManualSplitPrototype()

    with pytest.raises(RuntimeError, match="reset"):
        prototype.snapshot()


def test_manual_split_rejects_dataset_without_targets() -> None:
    """Manual split requires labels."""
    prototype = ManualSplitPrototype()
    dataset = DatasetSnapshot(
        features=np.array([[0.0, 0.0], [1.0, 1.0]]),
        targets=None,
    )

    with pytest.raises(ValueError, match="targets are required"):
        prototype.reset(dataset)


def test_manual_split_rejects_invalid_split_candidate() -> None:
    """Invalid split candidates should fail clearly."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )
    prototype.reset(_axis_dataset())

    with pytest.raises(ValueError, match="feature_index"):
        prototype.set_split(
            SplitCandidate(
                feature_index=10,
                threshold=0.0,
            ),
        )


def test_manual_split_rejects_empty_child_split() -> None:
    """Manual split must create two non-empty children."""
    prototype = ManualSplitPrototype(
        ManualSplitConfig(
            feature_index=0,
            threshold=0.0,
        ),
    )
    prototype.reset(_axis_dataset())

    with pytest.raises(ValueError, match="two non-empty children"):
        prototype.set_split(
            SplitCandidate(
                feature_index=0,
                threshold=100.0,
            ),
        )
