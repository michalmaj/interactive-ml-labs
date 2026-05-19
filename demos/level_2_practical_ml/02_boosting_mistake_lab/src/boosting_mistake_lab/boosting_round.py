"""Single boosting round utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import ArrayLike, NDArray

from boosting_mistake_lab.dataset import (
    WeightedDatasetSplit,
    WeightedTrainTestDataset,
)
from boosting_mistake_lab.weak_learner import (
    WeakLearnerBaseline,
    WeakLearnerConfig,
)

type FloatArray = NDArray[np.float64]

DEFAULT_ROUND_INDEX: Final[int] = 1
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 1
MIN_ROUND_INDEX: Final[int] = 1
MIN_SAMPLES_LEAF_LOWER_BOUND: Final[int] = 1
NORMALIZED_WEIGHT_SUM: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class BoostingRoundConfig:
    """Configuration for one boosting round.

    Attributes:
        round_index: One-based boosting round index.
        min_samples_leaf: Minimum number of samples required in each stump leaf.
    """

    round_index: int = DEFAULT_ROUND_INDEX
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF


@dataclass(frozen=True, slots=True)
class BoostingRoundResult:
    """Result of one boosting round.

    Attributes:
        round_index: One-based boosting round index.
        weak_learner: Fitted weak learner.
        weak_snapshot: Snapshot returned by the weak learner.
        round_snapshot: Snapshot summarizing the full boosting round.
        next_dataset: Dataset with updated training sample weights.
    """

    round_index: int
    weak_learner: WeakLearnerBaseline
    weak_snapshot: AlgorithmSnapshot
    round_snapshot: AlgorithmSnapshot
    next_dataset: WeightedTrainTestDataset

    @property
    def learner_weight(self) -> float:
        """Return learner weight alpha."""
        return float(self.round_snapshot.metrics["learner_weight"])

    @property
    def weighted_train_error(self) -> float:
        """Return weighted training error."""
        return float(self.round_snapshot.metrics["weighted_train_error"])

    @property
    def updated_train_weights(self) -> FloatArray:
        """Return updated train sample weights."""
        return np.asarray(
            self.round_snapshot.visual_state["updated_train_sample_weights"],
            dtype=float,
        )


def run_boosting_round(
    dataset: WeightedTrainTestDataset,
    config: BoostingRoundConfig | None = None,
) -> BoostingRoundResult:
    """Run one complete boosting round.

    Args:
        dataset: Weighted train/test dataset for the current round.
        config: Optional round configuration.

    Returns:
        BoostingRoundResult containing the fitted weak learner, round snapshot,
        and next-round dataset.
    """
    config = config or BoostingRoundConfig()
    _validate_config(config)

    weak_learner = WeakLearnerBaseline(
        WeakLearnerConfig(min_samples_leaf=config.min_samples_leaf),
    )
    weak_snapshot = weak_learner.reset(dataset)

    current_train_weights = np.asarray(dataset.train.sample_weights, dtype=float)
    updated_train_weights = np.asarray(
        weak_snapshot.visual_state["updated_train_sample_weights"],
        dtype=float,
    )

    next_dataset = make_next_round_dataset(
        dataset=dataset,
        updated_train_sample_weights=updated_train_weights,
        round_index=config.round_index,
    )
    round_snapshot = _build_round_snapshot(
        config=config,
        weak_snapshot=weak_snapshot,
        current_train_weights=current_train_weights,
        updated_train_weights=updated_train_weights,
    )

    return BoostingRoundResult(
        round_index=config.round_index,
        weak_learner=weak_learner,
        weak_snapshot=weak_snapshot,
        round_snapshot=round_snapshot,
        next_dataset=next_dataset,
    )


def make_next_round_dataset(
    *,
    dataset: WeightedTrainTestDataset,
    updated_train_sample_weights: ArrayLike,
    round_index: int,
) -> WeightedTrainTestDataset:
    """Create a next-round dataset with updated train sample weights.

    Args:
        dataset: Current weighted train/test dataset.
        updated_train_sample_weights: Updated normalized training sample weights.
        round_index: Round index that produced the updated weights.

    Returns:
        WeightedTrainTestDataset with updated train weights and unchanged test split.

    Raises:
        ValueError: If updated weights are invalid.
    """
    train_sample_count = np.asarray(dataset.train.snapshot.features).shape[0]
    updated_weights = _validate_sample_weights(
        sample_weights=updated_train_sample_weights,
        sample_count=train_sample_count,
        name="updated_train_sample_weights",
    )

    metadata = dict(dataset.metadata)
    metadata["last_completed_boosting_round"] = round_index
    metadata["train_sample_weight_sum"] = float(np.sum(updated_weights))

    return WeightedTrainTestDataset(
        train=WeightedDatasetSplit(
            snapshot=dataset.train.snapshot,
            sample_weights=updated_weights,
        ),
        test=dataset.test,
        metadata=metadata,
    )


def _build_round_snapshot(
    *,
    config: BoostingRoundConfig,
    weak_snapshot: AlgorithmSnapshot,
    current_train_weights: FloatArray,
    updated_train_weights: FloatArray,
) -> AlgorithmSnapshot:
    """Build an algorithm snapshot for one full boosting round."""
    weight_l1_change = float(np.sum(np.abs(updated_train_weights - current_train_weights)))

    return AlgorithmSnapshot(
        iteration=config.round_index,
        status="completed",
        metrics={
            "round_index": config.round_index,
            "train_accuracy": weak_snapshot.metrics["train_accuracy"],
            "test_accuracy": weak_snapshot.metrics["test_accuracy"],
            "train_error": weak_snapshot.metrics["train_error"],
            "test_error": weak_snapshot.metrics["test_error"],
            "weighted_train_error": weak_snapshot.metrics["weighted_train_error"],
            "weighted_train_accuracy": weak_snapshot.metrics["weighted_train_accuracy"],
            "weighted_test_error": weak_snapshot.metrics["weighted_test_error"],
            "weighted_test_accuracy": weak_snapshot.metrics["weighted_test_accuracy"],
            "learner_weight": weak_snapshot.metrics["learner_weight"],
            "feature_index": weak_snapshot.metrics["feature_index"],
            "threshold": weak_snapshot.metrics["threshold"],
            "old_mistake_weight_sum": weak_snapshot.metrics["old_mistake_weight_sum"],
            "updated_mistake_weight_sum": weak_snapshot.metrics["updated_mistake_weight_sum"],
            "old_correct_weight_sum": weak_snapshot.metrics["old_correct_weight_sum"],
            "updated_correct_weight_sum": weak_snapshot.metrics["updated_correct_weight_sum"],
            "min_current_train_weight": float(np.min(current_train_weights)),
            "max_current_train_weight": float(np.max(current_train_weights)),
            "min_updated_train_weight": float(np.min(updated_train_weights)),
            "max_updated_train_weight": float(np.max(updated_train_weights)),
            "weight_l1_change": weight_l1_change,
            "min_samples_leaf": config.min_samples_leaf,
        },
        visual_state={
            "weak_snapshot": weak_snapshot,
            "train_features": weak_snapshot.visual_state["train_features"],
            "train_targets": weak_snapshot.visual_state["train_targets"],
            "train_predictions": weak_snapshot.visual_state["train_predictions"],
            "train_mistakes": weak_snapshot.visual_state["train_mistakes"],
            "train_correct": weak_snapshot.visual_state["train_correct"],
            "current_train_sample_weights": current_train_weights,
            "updated_train_sample_weights": updated_train_weights,
            "test_features": weak_snapshot.visual_state["test_features"],
            "test_targets": weak_snapshot.visual_state["test_targets"],
            "test_predictions": weak_snapshot.visual_state["test_predictions"],
            "test_mistakes": weak_snapshot.visual_state["test_mistakes"],
            "split": weak_snapshot.visual_state["split"],
            "learner_weight_result": weak_snapshot.visual_state["learner_weight_result"],
            "weight_update_result": weak_snapshot.visual_state["weight_update_result"],
        },
        annotations=(
            f"Boosting round {config.round_index} completed.",
            "Weighted stump fitted, learner weight computed, and sample weights updated.",
            f"Weighted train error: {weak_snapshot.metrics['weighted_train_error']:.3f}.",
            f"Learner weight: {weak_snapshot.metrics['learner_weight']:.3f}.",
        ),
        done=True,
    )


def _validate_sample_weights(
    *,
    sample_weights: ArrayLike,
    sample_count: int,
    name: str,
) -> FloatArray:
    """Validate normalized sample weights."""
    weights = np.asarray(sample_weights, dtype=float)

    if weights.ndim != 1:
        msg = f"{name} must be a one-dimensional array."
        raise ValueError(msg)

    if weights.shape[0] != sample_count:
        msg = f"{name} must match sample count. Got {weights.shape[0]} and {sample_count}."
        raise ValueError(msg)

    if np.any(weights < 0.0):
        msg = f"{name} cannot contain negative values."
        raise ValueError(msg)

    if not np.isclose(np.sum(weights), NORMALIZED_WEIGHT_SUM):
        msg = f"{name} must sum to 1.0."
        raise ValueError(msg)

    return weights


def _validate_config(config: BoostingRoundConfig) -> None:
    """Validate boosting round configuration."""
    if config.round_index < MIN_ROUND_INDEX:
        msg = "round_index must be greater than or equal to 1."
        raise ValueError(msg)

    if config.min_samples_leaf < MIN_SAMPLES_LEAF_LOWER_BOUND:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)
