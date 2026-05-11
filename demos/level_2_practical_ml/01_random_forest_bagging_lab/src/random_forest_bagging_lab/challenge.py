"""Challenge mode for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from random_forest_bagging_lab.report import ModelComparisonReport

STATUS_SUCCESS: Final[str] = "success"
STATUS_FAILED: Final[str] = "failed"

DEFAULT_TARGET_TEST_ACCURACY: Final[float] = 0.90
DEFAULT_MAX_TREE_COUNT: Final[int] = 25
DEFAULT_MAX_GENERALIZATION_GAP: Final[float] = 0.15

MIN_TARGET_TEST_ACCURACY: Final[float] = 0.0
MAX_TARGET_TEST_ACCURACY: Final[float] = 1.0
MIN_TREE_COUNT: Final[int] = 1
MIN_GENERALIZATION_GAP: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class RandomForestChallengeConfig:
    """Configuration for Random Forest challenge mode.

    Attributes:
        target_test_accuracy: Minimum required forest test accuracy.
        max_tree_count: Maximum allowed number of trees.
        max_generalization_gap: Maximum allowed train-test accuracy gap.
    """

    target_test_accuracy: float = DEFAULT_TARGET_TEST_ACCURACY
    max_tree_count: int = DEFAULT_MAX_TREE_COUNT
    max_generalization_gap: float = DEFAULT_MAX_GENERALIZATION_GAP


@dataclass(frozen=True, slots=True)
class RandomForestChallengeResult:
    """Result of evaluating the current Random Forest challenge.

    Attributes:
        status: Challenge status.
        target_test_accuracy: Required forest test accuracy.
        max_tree_count: Maximum allowed tree count.
        max_generalization_gap: Maximum allowed generalization gap.
        forest_test_accuracy: Current forest test accuracy.
        forest_train_accuracy: Current forest train accuracy.
        forest_tree_count: Current number of trees.
        forest_generalization_gap: Current forest train-test gap.
        baseline_test_accuracy: Single-tree test accuracy.
        test_accuracy_delta: Forest test accuracy minus single-tree test accuracy.
        winner: Winner by test accuracy from the comparison report.
        message: Short student-facing explanation.
    """

    status: str
    target_test_accuracy: float
    max_tree_count: int
    max_generalization_gap: float
    forest_test_accuracy: float
    forest_train_accuracy: float
    forest_tree_count: int
    forest_generalization_gap: float
    baseline_test_accuracy: float
    test_accuracy_delta: float
    winner: str
    message: str

    @property
    def success(self) -> bool:
        """Return whether the challenge target is satisfied."""
        return self.status == STATUS_SUCCESS

    @property
    def failed(self) -> bool:
        """Return whether the challenge target is not satisfied."""
        return self.status == STATUS_FAILED


class RandomForestChallenge:
    """Evaluate Random Forest challenge status."""

    def __init__(self, config: RandomForestChallengeConfig | None = None) -> None:
        """Initialize the challenge."""
        self._config = config or RandomForestChallengeConfig()
        _validate_config(self._config)

    @property
    def config(self) -> RandomForestChallengeConfig:
        """Return challenge configuration."""
        return self._config

    def evaluate(self, report: ModelComparisonReport) -> RandomForestChallengeResult:
        """Evaluate the current comparison report against challenge rules.

        Args:
            report: Comparison between single-tree baseline and random forest.

        Returns:
            Challenge result.
        """
        forest_tree_count = _resolve_forest_tree_count(report)

        accuracy_ok = report.forest.test_accuracy >= self._config.target_test_accuracy
        tree_count_ok = forest_tree_count <= self._config.max_tree_count
        gap_ok = report.forest.generalization_gap <= self._config.max_generalization_gap

        if accuracy_ok and tree_count_ok and gap_ok:
            return RandomForestChallengeResult(
                status=STATUS_SUCCESS,
                target_test_accuracy=self._config.target_test_accuracy,
                max_tree_count=self._config.max_tree_count,
                max_generalization_gap=self._config.max_generalization_gap,
                forest_test_accuracy=report.forest.test_accuracy,
                forest_train_accuracy=report.forest.train_accuracy,
                forest_tree_count=forest_tree_count,
                forest_generalization_gap=report.forest.generalization_gap,
                baseline_test_accuracy=report.single_tree.test_accuracy,
                test_accuracy_delta=report.test_accuracy_delta,
                winner=report.winner,
                message=(
                    "Challenge completed: forest generalizes well without using too many trees."
                ),
            )

        return RandomForestChallengeResult(
            status=STATUS_FAILED,
            target_test_accuracy=self._config.target_test_accuracy,
            max_tree_count=self._config.max_tree_count,
            max_generalization_gap=self._config.max_generalization_gap,
            forest_test_accuracy=report.forest.test_accuracy,
            forest_train_accuracy=report.forest.train_accuracy,
            forest_tree_count=forest_tree_count,
            forest_generalization_gap=report.forest.generalization_gap,
            baseline_test_accuracy=report.single_tree.test_accuracy,
            test_accuracy_delta=report.test_accuracy_delta,
            winner=report.winner,
            message=_build_failure_message(
                accuracy_ok=accuracy_ok,
                tree_count_ok=tree_count_ok,
                gap_ok=gap_ok,
            ),
        )


def _resolve_forest_tree_count(report: ModelComparisonReport) -> int:
    """Read forest tree count from a comparison report."""
    if report.forest.tree_count is None:
        msg = "Forest tree_count is required for challenge evaluation."
        raise ValueError(msg)

    return report.forest.tree_count


def _build_failure_message(
    *,
    accuracy_ok: bool,
    tree_count_ok: bool,
    gap_ok: bool,
) -> str:
    """Build a concise failure explanation."""
    failed_reasons: list[str] = []

    if not accuracy_ok:
        failed_reasons.append("test accuracy is too low")

    if not tree_count_ok:
        failed_reasons.append("too many trees are used")

    if not gap_ok:
        failed_reasons.append("generalization gap is too high")

    return "Challenge not completed: " + ", ".join(failed_reasons) + "."


def _validate_config(config: RandomForestChallengeConfig) -> None:
    """Validate challenge configuration."""
    if not MIN_TARGET_TEST_ACCURACY < config.target_test_accuracy <= MAX_TARGET_TEST_ACCURACY:
        msg = "target_test_accuracy must be in the range (0, 1]."
        raise ValueError(msg)

    if config.max_tree_count < MIN_TREE_COUNT:
        msg = "max_tree_count must be greater than or equal to 1."
        raise ValueError(msg)

    if config.max_generalization_gap < MIN_GENERALIZATION_GAP:
        msg = "max_generalization_gap cannot be negative."
        raise ValueError(msg)
