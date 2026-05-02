"""Tests for shared snapshot objects."""

from dataclasses import FrozenInstanceError

import pytest
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot


def test_algorithm_snapshot_has_safe_defaults() -> None:
    """AlgorithmSnapshot should provide useful default values."""
    snapshot = AlgorithmSnapshot()

    assert snapshot.iteration == 0
    assert snapshot.status == "initialized"
    assert snapshot.visual_state == {}
    assert snapshot.metrics == {}
    assert snapshot.annotations == ()
    assert snapshot.done is False
    assert snapshot.success is None


def test_algorithm_snapshot_is_frozen() -> None:
    """AlgorithmSnapshot should prevent accidental field reassignment."""
    snapshot = AlgorithmSnapshot()

    with pytest.raises(FrozenInstanceError):
        snapshot.iteration = 1  # type: ignore[misc]


def test_dataset_snapshot_stores_dataset_information() -> None:
    """DatasetSnapshot should store basic dataset metadata."""
    snapshot = DatasetSnapshot(
        features=[[1.0, 2.0], [3.0, 4.0]],
        targets=[0, 1],
        feature_names=("x1", "x2"),
        target_names=("negative", "positive"),
        metadata={"source": "synthetic"},
    )

    assert snapshot.features == [[1.0, 2.0], [3.0, 4.0]]
    assert snapshot.targets == [0, 1]
    assert snapshot.feature_names == ("x1", "x2")
    assert snapshot.target_names == ("negative", "positive")
    assert snapshot.metadata["source"] == "synthetic"
