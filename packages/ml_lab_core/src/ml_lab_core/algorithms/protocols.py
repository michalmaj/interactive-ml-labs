"""Protocols for stepwise machine learning algorithms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ml_lab_core.snapshots import AlgorithmSnapshot, DatasetSnapshot


@runtime_checkable
class StepwiseAlgorithm(Protocol):
    """Protocol for algorithms that can be executed step by step.

    A stepwise algorithm exposes its progress through snapshots. This allows
    demos to animate learning or inference one logical operation at a time.

    Examples of one logical step:

    - one gradient descent update,
    - one k-means assignment or centroid update,
    - one decision-tree split,
    - one Q-learning transition.
    """

    name: str

    def reset(self, dataset: DatasetSnapshot) -> None:
        """Reset the algorithm using the provided dataset."""

    def step(self) -> AlgorithmSnapshot:
        """Perform one algorithmic step and return the new snapshot."""

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current algorithm state without modifying it."""
