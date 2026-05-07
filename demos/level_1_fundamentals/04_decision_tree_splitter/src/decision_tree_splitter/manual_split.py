"""Manual split prototype for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from numpy.typing import NDArray

from decision_tree_splitter.split import (
    CRITERION_GINI,
    ImpurityCriterion,
    SplitCandidate,
    SplitEvaluation,
    evaluate_split,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type BoolArray = NDArray[np.bool_]

DEFAULT_FEATURE_INDEX: Final[int] = 0
DEFAULT_THRESHOLD: Final[float] = 0.0
DEFAULT_CRITERION: Final[str] = CRITERION_GINI


@dataclass(frozen=True, slots=True)
class ManualSplitConfig:
    """Configuration for manual split evaluation.

    Attributes:
        feature_index: Index of the feature used for the initial split.
        threshold: Initial split threshold.
        criterion: Impurity criterion used for evaluating the split.
    """

    feature_index: int = DEFAULT_FEATURE_INDEX
    threshold: float = DEFAULT_THRESHOLD
    criterion: ImpurityCriterion = DEFAULT_CRITERION


class ManualSplitPrototype:
    """Evaluate user-selected decision-tree splits step by step.

    This class does not train a full tree yet. It represents the future manual
    split mode: a student selects one axis-aligned split and the prototype
    reports how good that split is.
    """

    name: str = "manual_split_prototype"

    def __init__(self, config: ManualSplitConfig | None = None) -> None:
        """Initialize the manual split prototype.

        Args:
            config: Optional manual split configuration.
        """
        self._config = config or ManualSplitConfig()
        self._candidate = SplitCandidate(
            feature_index=self._config.feature_index,
            threshold=self._config.threshold,
        )
        self._criterion = self._config.criterion

        self._features: FloatArray | None = None
        self._targets: IntArray | None = None
        self._evaluation: SplitEvaluation | None = None
        self._iteration = 0

    @property
    def candidate(self) -> SplitCandidate:
        """Return the currently selected split candidate."""
        return self._candidate

    @property
    def criterion(self) -> str:
        """Return the active impurity criterion."""
        return self._criterion

    @property
    def evaluation(self) -> SplitEvaluation:
        """Return the current split evaluation.

        Raises:
            RuntimeError: If the prototype has not been reset yet.
        """
        if self._evaluation is None:
            msg = "The prototype must be reset with a dataset before reading evaluation."
            raise RuntimeError(msg)

        return self._evaluation

    def reset(self, dataset: DatasetSnapshot) -> AlgorithmSnapshot:
        """Reset the prototype using a dataset snapshot.

        Args:
            dataset: Dataset containing features and binary labels.

        Returns:
            Initial algorithm snapshot.

        Raises:
            ValueError: If the dataset or configured split is invalid.
        """
        if dataset.targets is None:
            msg = "Dataset targets are required for manual split evaluation."
            raise ValueError(msg)

        self._features = np.asarray(dataset.features, dtype=float)
        self._targets = np.asarray(dataset.targets, dtype=int)
        self._iteration = 0
        self._evaluation = self._evaluate_current_candidate()

        return self.snapshot()

    def set_split(
        self,
        candidate: SplitCandidate,
        *,
        criterion: ImpurityCriterion | None = None,
    ) -> AlgorithmSnapshot:
        """Set a new manual split and evaluate it.

        Args:
            candidate: New split candidate.
            criterion: Optional impurity criterion. If omitted, the current
                criterion is preserved.

        Returns:
            Updated algorithm snapshot.

        Raises:
            RuntimeError: If the prototype has not been reset yet.
            ValueError: If the candidate or criterion is invalid.
        """
        self._ensure_initialized()

        self._candidate = candidate

        if criterion is not None:
            self._criterion = criterion

        self._evaluation = self._evaluate_current_candidate()
        self._iteration += 1

        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current manual split state."""
        self._ensure_initialized()

        assert self._features is not None
        assert self._targets is not None
        assert self._evaluation is not None

        left_mask = self._features[:, self._candidate.feature_index] <= self._candidate.threshold
        right_mask = ~left_mask

        return AlgorithmSnapshot(
            iteration=self._iteration,
            status="ready",
            visual_state={
                "features": self._features,
                "targets": self._targets,
                "left_mask": left_mask,
                "right_mask": right_mask,
                "candidate": self._candidate,
                "criterion": self._criterion,
            },
            metrics={
                "feature_index": self._candidate.feature_index,
                "threshold": self._candidate.threshold,
                "criterion": self._criterion,
                "parent_impurity": self._evaluation.parent_impurity,
                "left_impurity": self._evaluation.left_impurity,
                "right_impurity": self._evaluation.right_impurity,
                "weighted_child_impurity": self._evaluation.weighted_child_impurity,
                "information_gain": self._evaluation.information_gain,
                "left_sample_count": self._evaluation.left_sample_count,
                "right_sample_count": self._evaluation.right_sample_count,
            },
            annotations=(
                _build_split_rule(self._candidate),
                _build_gain_annotation(self._evaluation),
            ),
            done=True,
        )

    def _evaluate_current_candidate(self) -> SplitEvaluation:
        """Evaluate the current split candidate."""
        assert self._features is not None
        assert self._targets is not None

        return evaluate_split(
            self._features,
            self._targets,
            self._candidate,
            criterion=self._criterion,
        )

    def _ensure_initialized(self) -> None:
        """Ensure that the prototype was initialized with a dataset."""
        if self._features is None or self._targets is None or self._evaluation is None:
            msg = "The prototype must be reset with a dataset before use."
            raise RuntimeError(msg)


def _build_split_rule(candidate: SplitCandidate) -> str:
    """Build a short human-readable split rule."""
    return f"Manual split: x{candidate.feature_index + 1} <= {candidate.threshold:.3f}."


def _build_gain_annotation(evaluation: SplitEvaluation) -> str:
    """Build a short human-readable information gain annotation."""
    return f"Information gain: {evaluation.information_gain:.4f} using {evaluation.criterion}."
