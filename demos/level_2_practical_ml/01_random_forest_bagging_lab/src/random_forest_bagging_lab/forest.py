"""Random forest model for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from decision_tree_splitter import (
    DecisionTreeConfig,
    RecursiveDecisionTree,
)
from decision_tree_splitter.split import CRITERION_GINI, ImpurityCriterion
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import ArrayLike, NDArray

from random_forest_bagging_lab.bootstrap import (
    BootstrapSample,
    BootstrapSampleConfig,
    make_bootstrap_sample,
)
from random_forest_bagging_lab.dataset import TrainTestDataset
from random_forest_bagging_lab.voting import VotingResult, majority_vote

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_TREE_COUNT: Final[int] = 25
DEFAULT_MAX_DEPTH: Final[int] = 2
DEFAULT_CRITERION: Final[ImpurityCriterion] = CRITERION_GINI
DEFAULT_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 1.0
DEFAULT_SEED: Final[int] = 42
DEFAULT_MIN_SAMPLES_SPLIT: Final[int] = 2
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 1
DEFAULT_MIN_INFORMATION_GAIN: Final[float] = 0.0

MIN_TREE_COUNT: Final[int] = 1
MIN_MAX_DEPTH: Final[int] = 1
MIN_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 0.0
MAX_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 1.0
MIN_SAMPLES_SPLIT_LOWER_BOUND: Final[int] = 2
MIN_SAMPLES_LEAF_LOWER_BOUND: Final[int] = 1


@dataclass(frozen=True, slots=True)
class RandomForestConfig:
    """Configuration for a small educational random forest classifier.

    Attributes:
        tree_count: Number of trees in the forest.
        max_depth: Maximum depth of each individual decision tree.
        criterion: Impurity criterion used by every tree.
        bootstrap_sample_ratio: Fraction of training samples drawn for each tree.
        seed: Base seed used to generate per-tree bootstrap seeds.
        min_samples_split: Minimum number of samples required to split a node.
        min_samples_leaf: Minimum number of samples required in each child.
        min_information_gain: Minimum information gain required to accept a split.
    """

    tree_count: int = DEFAULT_TREE_COUNT
    max_depth: int = DEFAULT_MAX_DEPTH
    criterion: ImpurityCriterion = DEFAULT_CRITERION
    bootstrap_sample_ratio: float = DEFAULT_BOOTSTRAP_SAMPLE_RATIO
    seed: int = DEFAULT_SEED
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF
    min_information_gain: float = DEFAULT_MIN_INFORMATION_GAIN


@dataclass(frozen=True, slots=True)
class ForestTreeMember:
    """One fitted tree inside a random forest.

    Attributes:
        tree_index: Tree index inside the ensemble.
        tree: Fitted recursive decision tree.
        bootstrap_sample: Bootstrap sample used to train this tree.
        train_predictions: Predictions of this tree on the full training split.
        test_predictions: Predictions of this tree on the test split.
    """

    tree_index: int
    tree: RecursiveDecisionTree
    bootstrap_sample: BootstrapSample
    train_predictions: IntArray
    test_predictions: IntArray


class RandomForestModel:
    """Small random forest classifier for educational demos.

    The model trains several decision trees on bootstrap samples and combines
    their predictions with majority voting.
    """

    name: str = "random_forest_model"

    def __init__(self, config: RandomForestConfig | None = None) -> None:
        """Initialize the random forest model."""
        self._config = config or RandomForestConfig()
        _validate_config(self._config)

        self._members: tuple[ForestTreeMember, ...] = ()
        self._class_count: int | None = None
        self._snapshot: AlgorithmSnapshot | None = None
        self._train_voting_result: VotingResult | None = None
        self._test_voting_result: VotingResult | None = None

    @property
    def members(self) -> tuple[ForestTreeMember, ...]:
        """Return fitted forest members.

        Raises:
            RuntimeError: If the forest has not been fitted yet.
        """
        self._ensure_fitted()

        return self._members

    def reset(self, dataset: TrainTestDataset) -> AlgorithmSnapshot:
        """Fit the forest and evaluate it on train and test splits.

        Args:
            dataset: Train/test dataset.

        Returns:
            Algorithm snapshot containing forest metrics.

        Raises:
            ValueError: If dataset or configuration is invalid.
        """
        train_features, train_targets = _extract_split_arrays(
            features=dataset.train.features,
            targets=dataset.train.targets,
            split_name="train",
        )
        test_features, test_targets = _extract_split_arrays(
            features=dataset.test.features,
            targets=dataset.test.targets,
            split_name="test",
        )

        self._class_count = _resolve_class_count(train_targets, test_targets)

        members = self._fit_members(
            dataset=dataset,
            train_features=train_features,
            test_features=test_features,
        )
        self._members = tuple(members)

        train_tree_predictions = np.vstack(
            [member.train_predictions for member in self._members],
        )
        test_tree_predictions = np.vstack(
            [member.test_predictions for member in self._members],
        )

        train_voting_result = majority_vote(
            train_tree_predictions,
            class_count=self._class_count,
        )
        test_voting_result = majority_vote(
            test_tree_predictions,
            class_count=self._class_count,
        )

        train_accuracy = _accuracy_score(train_targets, train_voting_result.predictions)
        test_accuracy = _accuracy_score(test_targets, test_voting_result.predictions)

        self._train_voting_result = train_voting_result
        self._test_voting_result = test_voting_result
        self._snapshot = self._build_snapshot(
            dataset=dataset,
            train_targets=train_targets,
            test_targets=test_targets,
            train_tree_predictions=train_tree_predictions,
            test_tree_predictions=test_tree_predictions,
            train_voting_result=train_voting_result,
            test_voting_result=test_voting_result,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
        )

        return self._snapshot

    def snapshot(self) -> AlgorithmSnapshot:
        """Return current forest snapshot.

        Raises:
            RuntimeError: If the forest has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._snapshot is not None

        return self._snapshot

    def predict(self, features: ArrayLike) -> IntArray:
        """Predict labels using majority voting.

        Args:
            features: Feature matrix or one feature vector.

        Returns:
            Final forest predictions.

        Raises:
            RuntimeError: If the forest has not been fitted yet.
        """
        return self.predict_with_confidence(features).predictions

    def predict_with_confidence(self, features: ArrayLike) -> VotingResult:
        """Predict labels and vote confidence using all fitted trees.

        Args:
            features: Feature matrix or one feature vector.

        Returns:
            Voting result containing predictions, confidence, and vote counts.

        Raises:
            RuntimeError: If the forest has not been fitted yet.
        """
        self._ensure_fitted()

        assert self._class_count is not None

        tree_predictions = np.vstack(
            [member.tree.predict(features) for member in self._members],
        )

        return majority_vote(tree_predictions, class_count=self._class_count)

    def _fit_members(
        self,
        *,
        dataset: TrainTestDataset,
        train_features: FloatArray,
        test_features: FloatArray,
    ) -> list[ForestTreeMember]:
        """Fit all forest members."""
        members: list[ForestTreeMember] = []

        for tree_index in range(self._config.tree_count):
            bootstrap_sample = make_bootstrap_sample(
                dataset.train,
                BootstrapSampleConfig(
                    sample_ratio=self._config.bootstrap_sample_ratio,
                    seed=self._config.seed + tree_index,
                ),
            )
            tree = RecursiveDecisionTree(
                DecisionTreeConfig(
                    criterion=self._config.criterion,
                    max_depth=self._config.max_depth,
                    min_samples_split=self._config.min_samples_split,
                    min_samples_leaf=self._config.min_samples_leaf,
                    min_information_gain=self._config.min_information_gain,
                ),
            )
            tree.reset(bootstrap_sample.dataset)

            members.append(
                ForestTreeMember(
                    tree_index=tree_index,
                    tree=tree,
                    bootstrap_sample=bootstrap_sample,
                    train_predictions=tree.predict(train_features),
                    test_predictions=tree.predict(test_features),
                ),
            )

        return members

    def _build_snapshot(
        self,
        *,
        dataset: TrainTestDataset,
        train_targets: IntArray,
        test_targets: IntArray,
        train_tree_predictions: IntArray,
        test_tree_predictions: IntArray,
        train_voting_result: VotingResult,
        test_voting_result: VotingResult,
        train_accuracy: float,
        test_accuracy: float,
    ) -> AlgorithmSnapshot:
        """Build a forest algorithm snapshot."""
        unique_sample_counts = np.array(
            [member.bootstrap_sample.unique_sample_count for member in self._members],
            dtype=float,
        )
        oob_sample_counts = np.array(
            [member.bootstrap_sample.oob_indices.shape[0] for member in self._members],
            dtype=float,
        )

        mean_unique_sample_count = float(np.mean(unique_sample_counts))
        mean_oob_sample_count = float(np.mean(oob_sample_counts))
        mean_train_confidence = float(np.mean(train_voting_result.confidence))
        mean_test_confidence = float(np.mean(test_voting_result.confidence))

        return AlgorithmSnapshot(
            iteration=self._config.tree_count,
            status="fitted",
            visual_state={
                "train_features": dataset.train.features,
                "train_targets": train_targets,
                "test_features": dataset.test.features,
                "test_targets": test_targets,
                "train_predictions": train_voting_result.predictions,
                "test_predictions": test_voting_result.predictions,
                "train_confidence": train_voting_result.confidence,
                "test_confidence": test_voting_result.confidence,
                "train_vote_counts": train_voting_result.vote_counts,
                "test_vote_counts": test_voting_result.vote_counts,
                "train_tree_predictions": train_tree_predictions,
                "test_tree_predictions": test_tree_predictions,
                "members": self._members,
            },
            metrics={
                "tree_count": self._config.tree_count,
                "max_depth": self._config.max_depth,
                "criterion": self._config.criterion,
                "bootstrap_sample_ratio": self._config.bootstrap_sample_ratio,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "mean_train_confidence": mean_train_confidence,
                "mean_test_confidence": mean_test_confidence,
                "mean_unique_sample_count": mean_unique_sample_count,
                "mean_oob_sample_count": mean_oob_sample_count,
            },
            annotations=(
                f"Random forest fitted with {self._config.tree_count} trees.",
                f"Train accuracy: {train_accuracy:.3f}; test accuracy: {test_accuracy:.3f}.",
            ),
            done=True,
        )

    def _ensure_fitted(self) -> None:
        """Ensure that the forest has been fitted."""
        if not self._members or self._snapshot is None:
            msg = "The random forest must be reset with a dataset before use."
            raise RuntimeError(msg)


def _extract_split_arrays(
    *,
    features: object,
    targets: object,
    split_name: str,
) -> tuple[FloatArray, IntArray]:
    """Extract and validate one dataset split."""
    if targets is None:
        msg = f"{split_name} targets are required for random forest evaluation."
        raise ValueError(msg)

    feature_values = np.asarray(features, dtype=float)
    target_values = np.asarray(targets)

    if feature_values.ndim != 2:
        msg = f"{split_name} features must be a two-dimensional array."
        raise ValueError(msg)

    if feature_values.shape[0] == 0:
        msg = f"{split_name} features cannot be empty."
        raise ValueError(msg)

    if target_values.ndim != 1:
        msg = f"{split_name} targets must be a one-dimensional array."
        raise ValueError(msg)

    if not np.issubdtype(target_values.dtype, np.integer):
        msg = f"{split_name} targets must contain integers."
        raise ValueError(msg)

    integer_targets = target_values.astype(int)

    if feature_values.shape[0] != integer_targets.shape[0]:
        msg = (
            f"{split_name} features and targets must contain the same number of samples. "
            f"Got {feature_values.shape[0]} and {integer_targets.shape[0]}."
        )
        raise ValueError(msg)

    if np.any(integer_targets < 0):
        msg = f"{split_name} targets cannot contain negative values."
        raise ValueError(msg)

    return feature_values, integer_targets


def _resolve_class_count(train_targets: IntArray, test_targets: IntArray) -> int:
    """Resolve number of classes from train and test labels."""
    max_label = max(int(np.max(train_targets)), int(np.max(test_targets)))

    return max_label + 1


def _accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    """Compute classification accuracy."""
    if y_true.shape[0] != y_pred.shape[0]:
        msg = (
            "y_true and y_pred must contain the same number of samples. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )
        raise ValueError(msg)

    return float(np.mean(y_true == y_pred))


def _validate_config(config: RandomForestConfig) -> None:
    """Validate random forest configuration."""
    if config.tree_count < MIN_TREE_COUNT:
        msg = "tree_count must be greater than or equal to 1."
        raise ValueError(msg)

    if config.max_depth < MIN_MAX_DEPTH:
        msg = "max_depth must be greater than or equal to 1."
        raise ValueError(msg)

    if config.criterion not in {"gini", "entropy"}:
        msg = "criterion must be one of: entropy, gini."
        raise ValueError(msg)

    if not MIN_BOOTSTRAP_SAMPLE_RATIO < config.bootstrap_sample_ratio <= MAX_BOOTSTRAP_SAMPLE_RATIO:
        msg = "bootstrap_sample_ratio must be in the range (0, 1]."
        raise ValueError(msg)

    if config.min_samples_split < MIN_SAMPLES_SPLIT_LOWER_BOUND:
        msg = "min_samples_split must be greater than or equal to 2."
        raise ValueError(msg)

    if config.min_samples_leaf < MIN_SAMPLES_LEAF_LOWER_BOUND:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)

    if config.min_information_gain < 0.0:
        msg = "min_information_gain cannot be negative."
        raise ValueError(msg)
