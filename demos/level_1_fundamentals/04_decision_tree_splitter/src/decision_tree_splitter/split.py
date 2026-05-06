"""Split scoring utilities for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from decision_tree_splitter.impurity import entropy_impurity, gini_impurity

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type ImpurityCriterion = Literal["gini", "entropy"]

CRITERION_GINI: Final[str] = "gini"
CRITERION_ENTROPY: Final[str] = "entropy"
VALID_CRITERIA: Final[frozenset[str]] = frozenset(
    {
        CRITERION_GINI,
        CRITERION_ENTROPY,
    },
)

EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_LABEL_DIMENSIONS: Final[int] = 1
MIN_FEATURE_COUNT: Final[int] = 1
MIN_SAMPLE_COUNT_FOR_SPLIT: Final[int] = 2
MIN_UNIQUE_VALUES_FOR_SPLIT: Final[int] = 2
LEFT_CHILD_INDEX: Final[int] = 0
RIGHT_CHILD_INDEX: Final[int] = 1


@dataclass(frozen=True, slots=True)
class SplitCandidate:
    """Candidate axis-aligned split.

    Attributes:
        feature_index: Index of the feature used for splitting.
        threshold: Threshold used in the rule `feature <= threshold`.
    """

    feature_index: int
    threshold: float


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Evaluation result for one split candidate.

    Attributes:
        candidate: Evaluated split candidate.
        criterion: Impurity criterion used for scoring.
        parent_impurity: Impurity before applying the split.
        left_impurity: Impurity of the left child.
        right_impurity: Impurity of the right child.
        weighted_child_impurity: Weighted average impurity after the split.
        information_gain: Reduction in impurity caused by the split.
        left_sample_count: Number of samples in the left child.
        right_sample_count: Number of samples in the right child.
    """

    candidate: SplitCandidate
    criterion: str
    parent_impurity: float
    left_impurity: float
    right_impurity: float
    weighted_child_impurity: float
    information_gain: float
    left_sample_count: int
    right_sample_count: int


def generate_split_candidates(features: ArrayLike) -> tuple[SplitCandidate, ...]:
    """Generate axis-aligned split candidates from feature values.

    Candidate thresholds are midpoints between consecutive unique values for
    each feature.

    Args:
        features: Two-dimensional feature matrix.

    Returns:
        Tuple of candidate splits.

    Raises:
        ValueError: If features are invalid.
    """
    feature_values = _validate_features(features)
    candidates: list[SplitCandidate] = []

    for feature_index in range(feature_values.shape[1]):
        unique_values = np.unique(feature_values[:, feature_index])

        if unique_values.size < MIN_UNIQUE_VALUES_FOR_SPLIT:
            continue

        thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

        candidates.extend(
            SplitCandidate(
                feature_index=feature_index,
                threshold=float(threshold),
            )
            for threshold in thresholds
        )

    return tuple(candidates)


def evaluate_split(
    features: ArrayLike,
    labels: ArrayLike,
    candidate: SplitCandidate,
    *,
    criterion: ImpurityCriterion = CRITERION_GINI,
) -> SplitEvaluation:
    """Evaluate one axis-aligned split candidate.

    Args:
        features: Two-dimensional feature matrix.
        labels: One-dimensional integer labels.
        candidate: Split candidate to evaluate.
        criterion: Impurity criterion: `gini` or `entropy`.

    Returns:
        SplitEvaluation containing impurity values and information gain.

    Raises:
        ValueError: If inputs are invalid or split creates an empty child.
    """
    feature_values, label_values = _validate_split_inputs(features, labels)
    _validate_candidate(candidate, feature_count=feature_values.shape[1])
    _validate_criterion(criterion)

    left_mask = feature_values[:, candidate.feature_index] <= candidate.threshold
    right_mask = ~left_mask

    left_sample_count = int(np.sum(left_mask))
    right_sample_count = int(np.sum(right_mask))

    if left_sample_count == 0 or right_sample_count == 0:
        msg = "split must create two non-empty children."
        raise ValueError(msg)

    class_count = int(np.max(label_values)) + 1

    parent_impurity = _compute_impurity(
        label_values,
        criterion=criterion,
        class_count=class_count,
    )
    left_impurity = _compute_impurity(
        label_values[left_mask],
        criterion=criterion,
        class_count=class_count,
    )
    right_impurity = _compute_impurity(
        label_values[right_mask],
        criterion=criterion,
        class_count=class_count,
    )

    sample_count = left_sample_count + right_sample_count

    weighted_child_impurity = (
        left_sample_count / sample_count * left_impurity
        + right_sample_count / sample_count * right_impurity
    )
    information_gain = parent_impurity - weighted_child_impurity

    return SplitEvaluation(
        candidate=candidate,
        criterion=criterion,
        parent_impurity=parent_impurity,
        left_impurity=left_impurity,
        right_impurity=right_impurity,
        weighted_child_impurity=weighted_child_impurity,
        information_gain=information_gain,
        left_sample_count=left_sample_count,
        right_sample_count=right_sample_count,
    )


def best_split(
    features: ArrayLike,
    labels: ArrayLike,
    *,
    criterion: ImpurityCriterion = CRITERION_GINI,
) -> SplitEvaluation:
    """Find the best axis-aligned split according to information gain.

    Args:
        features: Two-dimensional feature matrix.
        labels: One-dimensional integer labels.
        criterion: Impurity criterion: `gini` or `entropy`.

    Returns:
        Best split evaluation.

    Raises:
        ValueError: If inputs are invalid or no valid split exists.
    """
    feature_values, label_values = _validate_split_inputs(features, labels)
    _validate_criterion(criterion)

    candidates = generate_split_candidates(feature_values)
    evaluations: list[SplitEvaluation] = []

    for candidate in candidates:
        try:
            evaluations.append(
                evaluate_split(
                    feature_values,
                    label_values,
                    candidate,
                    criterion=criterion,
                ),
            )
        except ValueError:
            continue

    if not evaluations:
        msg = "No valid split candidates found."
        raise ValueError(msg)

    return max(
        evaluations,
        key=lambda evaluation: (
            evaluation.information_gain,
            -evaluation.candidate.feature_index,
            -evaluation.candidate.threshold,
        ),
    )


def _compute_impurity(
    labels: IntArray,
    *,
    criterion: str,
    class_count: int,
) -> float:
    """Compute impurity using selected criterion."""
    if criterion == CRITERION_GINI:
        return gini_impurity(labels, class_count=class_count)

    if criterion == CRITERION_ENTROPY:
        return entropy_impurity(labels, class_count=class_count)

    msg = f"Unsupported impurity criterion: {criterion}."
    raise ValueError(msg)


def _validate_split_inputs(features: ArrayLike, labels: ArrayLike) -> tuple[FloatArray, IntArray]:
    """Validate split feature matrix and labels."""
    feature_values = _validate_features(features)
    label_values = _validate_labels(labels)

    if feature_values.shape[0] != label_values.shape[0]:
        msg = (
            "features and labels must contain the same number of samples. "
            f"Got {feature_values.shape[0]} and {label_values.shape[0]}."
        )
        raise ValueError(msg)

    if feature_values.shape[0] < MIN_SAMPLE_COUNT_FOR_SPLIT:
        msg = "At least two samples are required to evaluate a split."
        raise ValueError(msg)

    return feature_values, label_values


def _validate_features(features: ArrayLike) -> FloatArray:
    """Validate feature matrix."""
    feature_values = np.asarray(features, dtype=float)

    if feature_values.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "features must be a two-dimensional array."
        raise ValueError(msg)

    if feature_values.shape[0] == 0:
        msg = "features cannot be empty."
        raise ValueError(msg)

    if feature_values.shape[1] < MIN_FEATURE_COUNT:
        msg = "features must contain at least one column."
        raise ValueError(msg)

    return feature_values


def _validate_labels(labels: ArrayLike) -> IntArray:
    """Validate class labels."""
    label_values = np.asarray(labels)

    if label_values.ndim != EXPECTED_LABEL_DIMENSIONS:
        msg = "labels must be a one-dimensional array."
        raise ValueError(msg)

    if label_values.size == 0:
        msg = "labels cannot be empty."
        raise ValueError(msg)

    if not np.issubdtype(label_values.dtype, np.integer):
        msg = "labels must contain integers."
        raise ValueError(msg)

    integer_labels = label_values.astype(int)

    if np.any(integer_labels < 0):
        msg = "labels cannot contain negative values."
        raise ValueError(msg)

    return integer_labels


def _validate_candidate(candidate: SplitCandidate, *, feature_count: int) -> None:
    """Validate split candidate."""
    if candidate.feature_index < 0:
        msg = "feature_index cannot be negative."
        raise ValueError(msg)

    if candidate.feature_index >= feature_count:
        msg = (
            "feature_index must be smaller than the number of features. "
            f"Got feature_index={candidate.feature_index} and feature_count={feature_count}."
        )
        raise ValueError(msg)

    if not np.isfinite(candidate.threshold):
        msg = "threshold must be finite."
        raise ValueError(msg)


def _validate_criterion(criterion: str) -> None:
    """Validate impurity criterion."""
    if criterion not in VALID_CRITERIA:
        msg = f"criterion must be one of: {', '.join(sorted(VALID_CRITERIA))}."
        raise ValueError(msg)
