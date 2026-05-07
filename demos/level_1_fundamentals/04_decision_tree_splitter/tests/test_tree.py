"""Tests for the recursive decision tree model."""

import numpy as np
import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    DecisionTreeConfig,
    DecisionTreeNode,
    RecursiveDecisionTree,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
PERFECT_ACCURACY: float = 1.0
STUMP_MAX_DEPTH: int = 1
XOR_SOLVING_MAX_DEPTH: int = 2
EXPECTED_XOR_NODE_COUNT: int = 7
EXPECTED_XOR_LEAF_COUNT: int = 4


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


def test_recursive_decision_tree_reset_returns_snapshot() -> None:
    """Reset should fit the tree and return a snapshot."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    snapshot = tree.reset(_axis_dataset())

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_recursive_decision_tree_solves_axis_aligned_dataset() -> None:
    """A shallow tree should solve low-noise axis-aligned data."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=STUMP_MAX_DEPTH),
    )

    snapshot = tree.reset(_axis_dataset())

    assert snapshot.metrics["training_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["actual_depth"] == STUMP_MAX_DEPTH


def test_recursive_decision_tree_solves_xor_with_depth_two() -> None:
    """A depth-2 tree should solve low-noise XOR data."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    snapshot = tree.reset(_xor_dataset())

    assert snapshot.metrics["training_accuracy"] == pytest.approx(PERFECT_ACCURACY)
    assert snapshot.metrics["actual_depth"] == XOR_SOLVING_MAX_DEPTH
    assert snapshot.metrics["node_count"] == EXPECTED_XOR_NODE_COUNT
    assert snapshot.metrics["leaf_count"] == EXPECTED_XOR_LEAF_COUNT


def test_recursive_decision_tree_depth_one_does_not_solve_xor() -> None:
    """A depth-1 tree should behave like a stump and fail on XOR."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=STUMP_MAX_DEPTH),
    )

    snapshot = tree.reset(_xor_dataset())

    assert float(snapshot.metrics["training_accuracy"]) < PERFECT_ACCURACY
    assert snapshot.metrics["actual_depth"] == STUMP_MAX_DEPTH


def test_recursive_decision_tree_snapshot_contains_visual_state() -> None:
    """Snapshot should expose data needed by future renderers."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    snapshot = tree.reset(_xor_dataset())

    assert "features" in snapshot.visual_state
    assert "targets" in snapshot.visual_state
    assert "predictions" in snapshot.visual_state
    assert "root" in snapshot.visual_state
    assert "nodes" in snapshot.visual_state


def test_recursive_decision_tree_exposes_root_node() -> None:
    """Fitted tree should expose the root node."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )

    tree.reset(_xor_dataset())

    assert isinstance(tree.root, DecisionTreeNode)
    assert tree.root.depth == 0
    assert tree.root.sample_count == SAMPLES_PER_CLASS * 2
    assert tree.root.left is not None
    assert tree.root.right is not None


def test_recursive_decision_tree_predicts_labels() -> None:
    """Fitted tree should predict labels for query points."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )
    tree.reset(_xor_dataset())

    predictions = tree.predict(
        np.array(
            [
                [-2.0, -2.0],
                [-2.0, 2.0],
                [2.0, -2.0],
                [2.0, 2.0],
            ],
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1, 0]))


def test_recursive_decision_tree_accepts_single_query_point() -> None:
    """Prediction should accept one query point."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )
    tree.reset(_xor_dataset())

    prediction = tree.predict([2.0, 2.0])

    assert prediction.shape == (1,)
    assert prediction[0] == 0


def test_recursive_decision_tree_supports_entropy_criterion() -> None:
    """Recursive tree should support entropy-based split selection."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(
            criterion="entropy",
            max_depth=XOR_SOLVING_MAX_DEPTH,
        ),
    )

    snapshot = tree.reset(_xor_dataset())

    assert snapshot.metrics["criterion"] == "entropy"
    assert snapshot.metrics["training_accuracy"] == pytest.approx(PERFECT_ACCURACY)


def test_recursive_decision_tree_step_returns_snapshot() -> None:
    """Step should return current fitted tree snapshot."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )
    tree.reset(_xor_dataset())

    snapshot = tree.step()

    assert snapshot.status == "fitted"
    assert snapshot.done is True


def test_recursive_decision_tree_rejects_usage_before_reset() -> None:
    """Using tree before reset should fail clearly."""
    tree = RecursiveDecisionTree()

    with pytest.raises(RuntimeError, match="reset"):
        tree.snapshot()


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            DecisionTreeConfig(max_depth=0),
            "max_depth",
        ),
        (
            DecisionTreeConfig(min_samples_split=1),
            "min_samples_split",
        ),
        (
            DecisionTreeConfig(min_samples_leaf=0),
            "min_samples_leaf",
        ),
        (
            DecisionTreeConfig(min_information_gain=-1.0),
            "min_information_gain",
        ),
    ],
)
def test_recursive_decision_tree_rejects_invalid_config(
    config: DecisionTreeConfig,
    expected_message: str,
) -> None:
    """Invalid configuration should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        RecursiveDecisionTree(config)


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
    ],
)
def test_recursive_decision_tree_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid datasets should fail clearly."""
    tree = RecursiveDecisionTree()

    with pytest.raises(ValueError, match=expected_message):
        tree.reset(dataset)


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
def test_recursive_decision_tree_rejects_invalid_prediction_features(
    features: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid prediction features should fail clearly."""
    tree = RecursiveDecisionTree(
        DecisionTreeConfig(max_depth=XOR_SOLVING_MAX_DEPTH),
    )
    tree.reset(_xor_dataset())

    with pytest.raises(ValueError, match=expected_message):
        tree.predict(features)
