"""Challenge mode for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ml_lab_core import AlgorithmSnapshot

from decision_tree_splitter.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
)

STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"

DEFAULT_TARGET_ACCURACY: Final[float] = 0.95
DEFAULT_AXIS_ALIGNED_MAX_DEPTH: Final[int] = 1
DEFAULT_XOR_MAX_DEPTH: Final[int] = 2


@dataclass(frozen=True, slots=True)
class DecisionTreeChallengeConfig:
    """Configuration for decision tree challenge mode.

    Attributes:
        target_accuracy: Minimum required training accuracy.
        axis_aligned_max_depth: Maximum allowed depth for the axis-aligned dataset.
        xor_max_depth: Maximum allowed depth for the XOR dataset.
    """

    target_accuracy: float = DEFAULT_TARGET_ACCURACY
    axis_aligned_max_depth: int = DEFAULT_AXIS_ALIGNED_MAX_DEPTH
    xor_max_depth: int = DEFAULT_XOR_MAX_DEPTH


@dataclass(frozen=True, slots=True)
class DecisionTreeChallengeResult:
    """Result of evaluating the current decision tree challenge.

    Attributes:
        status: Challenge status.
        dataset_kind: Dataset kind used by the current demo state.
        target_accuracy: Minimum required accuracy.
        max_allowed_depth: Maximum allowed configured tree depth.
        accuracy: Current training accuracy.
        max_depth: Current configured maximum tree depth.
        actual_depth: Actual fitted tree depth.
        node_count: Number of fitted tree nodes.
        leaf_count: Number of fitted tree leaves.
        message: Short student-facing explanation.
    """

    status: str
    dataset_kind: str
    target_accuracy: float
    max_allowed_depth: int
    accuracy: float
    max_depth: int
    actual_depth: int
    node_count: int
    leaf_count: int
    message: str

    @property
    def success(self) -> bool:
        """Return whether the challenge target is satisfied."""
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        """Return whether the challenge target is not satisfied."""
        return self.status == STATUS_FAILED


class DecisionTreeChallenge:
    """Evaluate decision tree challenge status for the current dataset."""

    def __init__(self, config: DecisionTreeChallengeConfig | None = None) -> None:
        """Initialize the challenge."""
        self._config = config or DecisionTreeChallengeConfig()
        _validate_config(self._config)

    @property
    def config(self) -> DecisionTreeChallengeConfig:
        """Return challenge configuration."""
        return self._config

    def evaluate(
        self,
        *,
        snapshot: AlgorithmSnapshot,
        dataset_kind: str,
    ) -> DecisionTreeChallengeResult:
        """Evaluate the current tree snapshot against the active challenge.

        Args:
            snapshot: Current recursive decision tree snapshot.
            dataset_kind: Current dataset kind.

        Returns:
            Challenge result with status and explanation.

        Raises:
            ValueError: If the dataset kind is unsupported.
        """
        max_allowed_depth = _max_allowed_depth_for_dataset(
            dataset_kind,
            config=self._config,
        )

        accuracy = float(snapshot.metrics["training_accuracy"])
        max_depth = int(snapshot.metrics["max_depth"])
        actual_depth = int(snapshot.metrics["actual_depth"])
        node_count = int(snapshot.metrics["node_count"])
        leaf_count = int(snapshot.metrics["leaf_count"])

        accuracy_ok = accuracy >= self._config.target_accuracy
        depth_ok = max_depth <= max_allowed_depth

        if accuracy_ok and depth_ok:
            return DecisionTreeChallengeResult(
                status=STATUS_SUCCESS,
                dataset_kind=dataset_kind,
                target_accuracy=self._config.target_accuracy,
                max_allowed_depth=max_allowed_depth,
                accuracy=accuracy,
                max_depth=max_depth,
                actual_depth=actual_depth,
                node_count=node_count,
                leaf_count=leaf_count,
                message="Challenge completed: the tree is accurate enough and simple enough.",
            )

        if not accuracy_ok and not depth_ok:
            message = "Challenge not completed: accuracy is too low and max_depth is too high."
        elif not accuracy_ok:
            message = "Challenge not completed: increase accuracy without making the tree too deep."
        else:
            message = "Challenge not completed: tree is accurate, but max_depth is too high."

        return DecisionTreeChallengeResult(
            status=STATUS_FAILED,
            dataset_kind=dataset_kind,
            target_accuracy=self._config.target_accuracy,
            max_allowed_depth=max_allowed_depth,
            accuracy=accuracy,
            max_depth=max_depth,
            actual_depth=actual_depth,
            node_count=node_count,
            leaf_count=leaf_count,
            message=message,
        )


def _max_allowed_depth_for_dataset(
    dataset_kind: str,
    *,
    config: DecisionTreeChallengeConfig,
) -> int:
    """Return challenge depth limit for a dataset kind."""
    if dataset_kind == DATASET_KIND_AXIS_ALIGNED:
        return config.axis_aligned_max_depth

    if dataset_kind == DATASET_KIND_XOR:
        return config.xor_max_depth

    msg = f"Unsupported dataset kind for challenge mode: {dataset_kind}."
    raise ValueError(msg)


def _validate_config(config: DecisionTreeChallengeConfig) -> None:
    """Validate decision tree challenge configuration."""
    if not 0.0 < config.target_accuracy <= 1.0:
        msg = "target_accuracy must be in the range (0, 1]."
        raise ValueError(msg)

    if config.axis_aligned_max_depth < 1:
        msg = "axis_aligned_max_depth must be greater than or equal to 1."
        raise ValueError(msg)

    if config.xor_max_depth < 1:
        msg = "xor_max_depth must be greater than or equal to 1."
        raise ValueError(msg)
