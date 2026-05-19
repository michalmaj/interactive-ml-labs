"""Multi-round boosting trainer for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import NDArray

from boosting_mistake_lab.boosting_round import (
    BoostingRoundConfig,
    BoostingRoundResult,
    run_boosting_round,
)
from boosting_mistake_lab.dataset import WeightedTrainTestDataset

type FloatArray = NDArray[np.float64]

DEFAULT_ROUND_COUNT: Final[int] = 5
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 1
MIN_ROUND_COUNT: Final[int] = 1
MIN_SAMPLES_LEAF_LOWER_BOUND: Final[int] = 1


@dataclass(frozen=True, slots=True)
class BoostingTrainerConfig:
    """Configuration for a multi-round boosting trainer.

    Attributes:
        round_count: Number of boosting rounds to run.
        min_samples_leaf: Minimum number of samples required in each stump leaf.
    """

    round_count: int = DEFAULT_ROUND_COUNT
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF


@dataclass(frozen=True, slots=True)
class BoostingTrainerResult:
    """Result of fitting multiple boosting rounds.

    Attributes:
        config: Trainer configuration.
        round_results: Results of all boosting rounds.
        initial_dataset: Dataset used at the beginning of training.
        final_dataset: Dataset with weights after the last boosting round.
        snapshot: Trainer-level snapshot.
    """

    config: BoostingTrainerConfig
    round_results: tuple[BoostingRoundResult, ...]
    initial_dataset: WeightedTrainTestDataset
    final_dataset: WeightedTrainTestDataset
    snapshot: AlgorithmSnapshot

    @property
    def round_count(self) -> int:
        """Return number of completed boosting rounds."""
        return len(self.round_results)

    @property
    def learner_weights(self) -> FloatArray:
        """Return learner weights for all completed rounds."""
        return np.asarray(
            [round_result.learner_weight for round_result in self.round_results],
            dtype=float,
        )

    @property
    def weighted_train_errors(self) -> FloatArray:
        """Return weighted training errors for all completed rounds."""
        return np.asarray(
            [round_result.weighted_train_error for round_result in self.round_results],
            dtype=float,
        )

    @property
    def final_train_weights(self) -> FloatArray:
        """Return final normalized train sample weights."""
        return np.asarray(self.final_dataset.train.sample_weights, dtype=float)


class BoostingTrainer:
    """Fit multiple boosting rounds sequentially.

    Each round receives the dataset returned by the previous round. This means
    that every new weak learner sees updated sample weights.
    """

    name: str = "boosting_trainer"

    def __init__(self, config: BoostingTrainerConfig | None = None) -> None:
        """Initialize the boosting trainer."""
        self._config = config or BoostingTrainerConfig()
        _validate_config(self._config)

        self._result: BoostingTrainerResult | None = None

    @property
    def config(self) -> BoostingTrainerConfig:
        """Return trainer configuration."""
        return self._config

    def reset(self, dataset: WeightedTrainTestDataset) -> BoostingTrainerResult:
        """Run all configured boosting rounds.

        Args:
            dataset: Initial weighted train/test dataset.

        Returns:
            BoostingTrainerResult with all round results and final dataset.
        """
        current_dataset = dataset
        round_results: list[BoostingRoundResult] = []

        for round_index in range(1, self._config.round_count + 1):
            round_result = run_boosting_round(
                current_dataset,
                BoostingRoundConfig(
                    round_index=round_index,
                    min_samples_leaf=self._config.min_samples_leaf,
                ),
            )
            round_results.append(round_result)
            current_dataset = round_result.next_dataset

        completed_rounds = tuple(round_results)
        snapshot = _build_trainer_snapshot(
            config=self._config,
            initial_dataset=dataset,
            final_dataset=current_dataset,
            round_results=completed_rounds,
        )

        self._result = BoostingTrainerResult(
            config=self._config,
            round_results=completed_rounds,
            initial_dataset=dataset,
            final_dataset=current_dataset,
            snapshot=snapshot,
        )

        return self._result

    def result(self) -> BoostingTrainerResult:
        """Return latest trainer result.

        Raises:
            RuntimeError: If trainer has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._result is not None

        return self._result

    def snapshot(self) -> AlgorithmSnapshot:
        """Return latest trainer snapshot.

        Raises:
            RuntimeError: If trainer has not been fitted yet.
        """
        return self.result().snapshot

    def _ensure_fitted(self) -> None:
        """Ensure that the trainer has been fitted."""
        if self._result is None:
            msg = "The boosting trainer must be reset with a dataset before use."
            raise RuntimeError(msg)


def _build_trainer_snapshot(
    *,
    config: BoostingTrainerConfig,
    initial_dataset: WeightedTrainTestDataset,
    final_dataset: WeightedTrainTestDataset,
    round_results: tuple[BoostingRoundResult, ...],
) -> AlgorithmSnapshot:
    """Build a trainer-level snapshot."""
    learner_weights = np.asarray(
        [round_result.learner_weight for round_result in round_results],
        dtype=float,
    )
    weighted_train_errors = np.asarray(
        [round_result.weighted_train_error for round_result in round_results],
        dtype=float,
    )
    weight_l1_changes = np.asarray(
        [
            float(round_result.round_snapshot.metrics["weight_l1_change"])
            for round_result in round_results
        ],
        dtype=float,
    )
    train_accuracies = np.asarray(
        [
            float(round_result.round_snapshot.metrics["train_accuracy"])
            for round_result in round_results
        ],
        dtype=float,
    )
    test_accuracies = np.asarray(
        [
            float(round_result.round_snapshot.metrics["test_accuracy"])
            for round_result in round_results
        ],
        dtype=float,
    )

    initial_train_weights = np.asarray(initial_dataset.train.sample_weights, dtype=float)
    final_train_weights = np.asarray(final_dataset.train.sample_weights, dtype=float)
    final_round = round_results[-1]

    return AlgorithmSnapshot(
        iteration=config.round_count,
        status="completed",
        metrics={
            "round_count": config.round_count,
            "completed_round_count": len(round_results),
            "final_round_index": final_round.round_index,
            "final_weighted_train_error": final_round.weighted_train_error,
            "final_learner_weight": final_round.learner_weight,
            "final_train_accuracy": float(
                final_round.round_snapshot.metrics["train_accuracy"],
            ),
            "final_test_accuracy": float(
                final_round.round_snapshot.metrics["test_accuracy"],
            ),
            "mean_weighted_train_error": float(np.mean(weighted_train_errors)),
            "mean_learner_weight": float(np.mean(learner_weights)),
            "min_learner_weight": float(np.min(learner_weights)),
            "max_learner_weight": float(np.max(learner_weights)),
            "cumulative_weight_l1_change": float(np.sum(weight_l1_changes)),
            "min_initial_train_weight": float(np.min(initial_train_weights)),
            "max_initial_train_weight": float(np.max(initial_train_weights)),
            "min_final_train_weight": float(np.min(final_train_weights)),
            "max_final_train_weight": float(np.max(final_train_weights)),
            "min_samples_leaf": config.min_samples_leaf,
        },
        visual_state={
            "round_snapshots": tuple(round_result.round_snapshot for round_result in round_results),
            "weak_snapshots": tuple(round_result.weak_snapshot for round_result in round_results),
            "staged_learner_weights": learner_weights,
            "staged_weighted_train_errors": weighted_train_errors,
            "staged_weight_l1_changes": weight_l1_changes,
            "staged_train_accuracies": train_accuracies,
            "staged_test_accuracies": test_accuracies,
            "initial_train_sample_weights": initial_train_weights,
            "final_train_sample_weights": final_train_weights,
            "final_dataset": final_dataset,
        },
        annotations=(
            f"Boosting trainer completed {config.round_count} rounds.",
            "Each round fitted a weighted stump and updated train sample weights.",
            f"Final weighted train error: {final_round.weighted_train_error:.3f}.",
            f"Final learner weight: {final_round.learner_weight:.3f}.",
        ),
        done=True,
    )


def _validate_config(config: BoostingTrainerConfig) -> None:
    """Validate boosting trainer configuration."""
    if config.round_count < MIN_ROUND_COUNT:
        msg = "round_count must be greater than or equal to 1."
        raise ValueError(msg)

    if config.min_samples_leaf < MIN_SAMPLES_LEAF_LOWER_BOUND:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)
