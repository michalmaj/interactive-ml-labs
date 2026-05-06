"""Tests for decision-tree split scoring."""

import numpy as np
import pytest
from decision_tree_splitter import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SplitCandidate,
    best_split,
    evaluate_split,
    generate_split_candidates,
)
from decision_tree_splitter.dataset import (
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)

EXPECTED_CANDIDATE_COUNT: int = 4
EXPECTED_PERFECT_GINI_GAIN: float = 0.5
EXPECTED_PERFECT_ENTROPY_GAIN: float = 1.0
SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123


def _simple_features() -> np.ndarray:
    """Create a tiny feature matrix with two columns."""
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=float,
    )


def test_generate_split_candidates_uses_midpoints_between_unique_values() -> None:
    """Candidate thresholds should be midpoints between unique values."""
    candidates = generate_split_candidates(_simple_features())

    assert candidates == (
        SplitCandidate(feature_index=0, threshold=0.5),
        SplitCandidate(feature_index=0, threshold=1.5),
        SplitCandidate(feature_index=1, threshold=0.5),
    )


def test_evaluate_split_computes_perfect_gini_gain() -> None:
    """A perfect binary split should reduce Gini from 0.5 to 0.0."""
    features = np.array(
        [
            [-1.0, 0.0],
            [-2.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])

    result = evaluate_split(
        features,
        labels,
        SplitCandidate(feature_index=0, threshold=0.0),
    )

    assert result.parent_impurity == pytest.approx(0.5)
    assert result.left_impurity == pytest.approx(0.0)
    assert result.right_impurity == pytest.approx(0.0)
    assert result.weighted_child_impurity == pytest.approx(0.0)
    assert result.information_gain == pytest.approx(EXPECTED_PERFECT_GINI_GAIN)
    assert result.left_sample_count == 2
    assert result.right_sample_count == 2


def test_evaluate_split_computes_perfect_entropy_gain() -> None:
    """A perfect binary split should reduce entropy from 1.0 to 0.0."""
    features = np.array(
        [
            [-1.0, 0.0],
            [-2.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1])

    result = evaluate_split(
        features,
        labels,
        SplitCandidate(feature_index=0, threshold=0.0),
        criterion="entropy",
    )

    assert result.parent_impurity == pytest.approx(1.0)
    assert result.left_impurity == pytest.approx(0.0)
    assert result.right_impurity == pytest.approx(0.0)
    assert result.weighted_child_impurity == pytest.approx(0.0)
    assert result.information_gain == pytest.approx(EXPECTED_PERFECT_ENTROPY_GAIN)


def test_evaluate_split_computes_non_perfect_gain() -> None:
    """A mixed child should produce lower information gain."""
    features = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=float,
    )
    labels = np.array([0, 1, 1, 1])

    result = evaluate_split(
        features,
        labels,
        SplitCandidate(feature_index=0, threshold=0.0),
    )

    assert result.parent_impurity == pytest.approx(0.375)
    assert result.left_impurity == pytest.approx(0.5)
    assert result.right_impurity == pytest.approx(0.0)
    assert result.weighted_child_impurity == pytest.approx(0.25)
    assert result.information_gain == pytest.approx(0.125)


def test_best_split_finds_axis_aligned_separation() -> None:
    """Best split should separate low-noise axis-aligned data."""
    dataset = make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(
            samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_AXIS_ALIGNED,
        ),
    )

    result = best_split(dataset.features, dataset.targets)

    assert result.candidate.feature_index == 0
    assert result.information_gain == pytest.approx(EXPECTED_PERFECT_GINI_GAIN)


def test_best_split_on_xor_is_not_perfect_at_root() -> None:
    """One root split should not perfectly solve XOR."""
    dataset = make_synthetic_decision_tree_dataset(
        SyntheticDecisionTreeDatasetConfig(
            samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_XOR,
        ),
    )

    result = best_split(dataset.features, dataset.targets)

    assert result.information_gain < EXPECTED_PERFECT_GINI_GAIN


@pytest.mark.parametrize(
    "features, expected_message",
    [
        (
            np.array([1.0, 2.0]),
            "two-dimensional",
        ),
        (
            np.empty((0, 2)),
            "features cannot be empty",
        ),
        (
            np.empty((2, 0)),
            "at least one column",
        ),
    ],
)
def test_generate_split_candidates_rejects_invalid_features(
    features: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid candidate-generation features should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        generate_split_candidates(features)


@pytest.mark.parametrize(
    "features, labels, expected_message",
    [
        (
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            "two-dimensional",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([[0]]),
            "one-dimensional",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([0.0]),
            "integers",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([-1]),
            "negative",
        ),
        (
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0]),
            "same number of samples",
        ),
        (
            np.array([[1.0, 2.0]]),
            np.array([0]),
            "At least two samples",
        ),
    ],
)
def test_evaluate_split_rejects_invalid_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid split inputs should fail clearly."""
    with pytest.raises(ValueError, match=expected_message):
        evaluate_split(
            features,
            labels,
            SplitCandidate(feature_index=0, threshold=0.0),
        )


@pytest.mark.parametrize(
    "candidate, expected_message",
    [
        (
            SplitCandidate(feature_index=-1, threshold=0.0),
            "feature_index cannot be negative",
        ),
        (
            SplitCandidate(feature_index=2, threshold=0.0),
            "feature_index must be smaller",
        ),
        (
            SplitCandidate(feature_index=0, threshold=float("inf")),
            "threshold must be finite",
        ),
    ],
)
def test_evaluate_split_rejects_invalid_candidate(
    candidate: SplitCandidate,
    expected_message: str,
) -> None:
    """Invalid split candidates should fail clearly."""
    features = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match=expected_message):
        evaluate_split(features, labels, candidate)


def test_evaluate_split_rejects_empty_child_split() -> None:
    """Split candidates must create two non-empty children."""
    features = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="two non-empty children"):
        evaluate_split(
            features,
            labels,
            SplitCandidate(feature_index=0, threshold=2.0),
        )


def test_best_split_rejects_dataset_without_valid_candidates() -> None:
    """Best split should fail when every feature has a constant value."""
    features = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="No valid split candidates"):
        best_split(features, labels)


def test_split_rejects_invalid_criterion() -> None:
    """Unsupported impurity criteria should fail clearly."""
    features = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="criterion"):
        evaluate_split(
            features,
            labels,
            SplitCandidate(feature_index=0, threshold=0.5),
            criterion="unknown",  # type: ignore[arg-type]
        )
