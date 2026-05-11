"""Majority voting utilities for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

type IntArray = NDArray[np.int_]
type FloatArray = NDArray[np.float64]

EXPECTED_PREDICTION_DIMENSIONS: Final[int] = 2
MIN_TREE_COUNT: Final[int] = 1
MIN_SAMPLE_COUNT: Final[int] = 1
MIN_CLASS_COUNT: Final[int] = 1


@dataclass(frozen=True, slots=True)
class VotingResult:
    """Result of majority voting over tree predictions.

    Attributes:
        predictions: Final majority-vote predictions for each sample.
        confidence: Vote confidence for each sample. This is the fraction of
            trees voting for the winning class.
        vote_counts: Matrix of class vote counts with shape
            `(sample_count, class_count)`.
    """

    predictions: IntArray
    confidence: FloatArray
    vote_counts: IntArray


def majority_vote(
    tree_predictions: ArrayLike,
    *,
    class_count: int | None = None,
) -> VotingResult:
    """Combine tree predictions using majority voting.

    Args:
        tree_predictions: Integer matrix with shape `(tree_count, sample_count)`.
            Each row contains predictions from one tree.
        class_count: Optional total number of classes. When provided, missing
            classes are represented with zero vote counts.

    Returns:
        VotingResult containing final predictions, vote confidence, and counts.

    Raises:
        ValueError: If predictions or class_count are invalid.
    """
    predictions = _validate_tree_predictions(tree_predictions)
    resolved_class_count = _resolve_class_count(predictions, class_count=class_count)

    vote_counts = _count_votes(
        predictions,
        class_count=resolved_class_count,
    )
    final_predictions = np.argmax(vote_counts, axis=1).astype(int)

    winning_votes = vote_counts[
        np.arange(vote_counts.shape[0]),
        final_predictions,
    ]
    confidence = winning_votes.astype(float) / predictions.shape[0]

    return VotingResult(
        predictions=final_predictions,
        confidence=confidence,
        vote_counts=vote_counts,
    )


def _count_votes(predictions: IntArray, *, class_count: int) -> IntArray:
    """Count votes for every sample and class."""
    sample_count = predictions.shape[1]
    vote_counts = np.zeros((sample_count, class_count), dtype=int)

    for sample_index in range(sample_count):
        vote_counts[sample_index] = np.bincount(
            predictions[:, sample_index],
            minlength=class_count,
        )

    return vote_counts


def _validate_tree_predictions(tree_predictions: ArrayLike) -> IntArray:
    """Validate and convert tree predictions."""
    predictions = np.asarray(tree_predictions)

    if predictions.ndim != EXPECTED_PREDICTION_DIMENSIONS:
        msg = "tree_predictions must be a two-dimensional array."
        raise ValueError(msg)

    if predictions.shape[0] < MIN_TREE_COUNT:
        msg = "tree_predictions must contain at least one tree."
        raise ValueError(msg)

    if predictions.shape[1] < MIN_SAMPLE_COUNT:
        msg = "tree_predictions must contain at least one sample."
        raise ValueError(msg)

    if not np.issubdtype(predictions.dtype, np.integer):
        msg = "tree_predictions must contain integers."
        raise ValueError(msg)

    integer_predictions = predictions.astype(int)

    if np.any(integer_predictions < 0):
        msg = "tree_predictions cannot contain negative class labels."
        raise ValueError(msg)

    return integer_predictions


def _resolve_class_count(predictions: IntArray, *, class_count: int | None) -> int:
    """Resolve and validate class count."""
    if class_count is None:
        return int(np.max(predictions)) + 1

    if class_count < MIN_CLASS_COUNT:
        msg = "class_count must be greater than 0."
        raise ValueError(msg)

    max_prediction = int(np.max(predictions))

    if max_prediction >= class_count:
        msg = (
            "class_count must be greater than the maximum predicted label. "
            f"Got class_count={class_count} and max_prediction={max_prediction}."
        )
        raise ValueError(msg)

    return class_count
