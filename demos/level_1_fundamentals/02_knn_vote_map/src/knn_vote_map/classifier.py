"""k-nearest neighbors classifier for the k-NN Vote Map demo."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot
from numpy.typing import ArrayLike, NDArray

from knn_vote_map.metrics import euclidean_distances

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

DEFAULT_K: Final[int] = 5
EXPECTED_FEATURE_DIMENSIONS: Final[int] = 2
EXPECTED_TARGET_DIMENSIONS: Final[int] = 1


@dataclass(frozen=True, slots=True)
class KNearestNeighborsConfig:
    """Configuration for the k-nearest neighbors classifier.

    Attributes:
        k: Number of nearest neighbors used for voting.
    """

    k: int = DEFAULT_K


@dataclass(frozen=True, slots=True)
class Neighbor:
    """Information about one nearest neighbor.

    Attributes:
        index: Index of the neighbor in the training dataset.
        label: Class label assigned to the neighbor.
        distance: Distance between the query point and the neighbor.
    """

    index: int
    label: int
    distance: float


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Result of one k-NN prediction.

    Attributes:
        predicted_label: Class selected by majority voting.
        query_point: Point that was classified.
        neighbors: Nearest neighbors used for voting.
        vote_counts: Mapping from class label to vote count.
    """

    predicted_label: int
    query_point: FloatArray
    neighbors: tuple[Neighbor, ...]
    vote_counts: dict[int, int]


class KNearestNeighborsClassifier:
    """Simple k-nearest neighbors classifier.

    The classifier stores training samples and predicts a label for a query
    point by majority voting among the nearest neighbors.
    """

    name: str = "knn_classifier"

    def __init__(self, config: KNearestNeighborsConfig | None = None) -> None:
        """Initialize the classifier with optional configuration."""
        self._config = config or KNearestNeighborsConfig()
        _validate_config(self._config)

        self._features: FloatArray | None = None
        self._targets: IntArray | None = None
        self._last_prediction: PredictionResult | None = None

    @property
    def k(self) -> int:
        """Return the number of neighbors used for voting."""
        return self._config.k

    @property
    def is_fitted(self) -> bool:
        """Return whether the classifier has training data."""
        return self._features is not None and self._targets is not None

    def fit(self, dataset: DatasetSnapshot) -> None:
        """Store training data from a dataset snapshot.

        Args:
            dataset: Dataset containing two-dimensional features and labels.

        Raises:
            ValueError: If the dataset has invalid shape or labels.
        """
        features, targets = _extract_classification_arrays(dataset)

        if self._config.k > features.shape[0]:
            msg = (
                "k cannot be greater than the number of training samples. "
                f"Got k={self._config.k} and sample_count={features.shape[0]}."
            )
            raise ValueError(msg)

        self._features = features
        self._targets = targets
        self._last_prediction = None

    def reset(self, dataset: DatasetSnapshot) -> None:
        """Reset classifier state using the provided dataset.

        This method mirrors the common algorithm interface used by other demos.
        """
        self.fit(dataset)

    def predict_one(self, query_point: ArrayLike) -> PredictionResult:
        """Predict the class label for one query point.

        Args:
            query_point: One-dimensional point with the same feature count as
                the training data.

        Returns:
            PredictionResult containing predicted class and voting details.

        Raises:
            RuntimeError: If the classifier has not been fitted yet.
            ValueError: If the query point shape is invalid.
        """
        self._ensure_fitted()

        assert self._features is not None
        assert self._targets is not None

        query = np.asarray(query_point, dtype=float)
        _validate_query_point(query, expected_feature_count=self._features.shape[1])

        distances = euclidean_distances(self._features, query)
        neighbor_indices = np.argsort(distances)[: self._config.k]

        neighbors = tuple(
            Neighbor(
                index=int(index),
                label=int(self._targets[index]),
                distance=float(distances[index]),
            )
            for index in neighbor_indices
        )

        vote_counts = _count_votes(neighbors)
        predicted_label = _select_winning_label(vote_counts)

        result = PredictionResult(
            predicted_label=predicted_label,
            query_point=query,
            neighbors=neighbors,
            vote_counts=vote_counts,
        )

        self._last_prediction = result

        return result

    def predict(self, query_points: ArrayLike) -> IntArray:
        """Predict labels for many query points.

        Args:
            query_points: Two-dimensional array of query points.

        Returns:
            One-dimensional array of predicted class labels.
        """
        self._ensure_fitted()

        values = np.asarray(query_points, dtype=float)

        if values.ndim == 1:
            result = self.predict_one(values)
            return np.array([result.predicted_label], dtype=int)

        if values.ndim != EXPECTED_FEATURE_DIMENSIONS:
            msg = "query_points must be a one-dimensional point or a two-dimensional array."
            raise ValueError(msg)

        predictions = [self.predict_one(query_point).predicted_label for query_point in values]

        return np.asarray(predictions, dtype=int)

    def step(self) -> AlgorithmSnapshot:
        """Return a snapshot of the latest prediction.

        k-NN does not perform iterative training. This method exists to keep the
        classifier compatible with demos that expect a stepwise-style object.
        """
        self._ensure_fitted()

        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current classifier state."""
        self._ensure_fitted()

        assert self._features is not None
        assert self._targets is not None

        visual_state: dict[str, object] = {
            "features": self._features,
            "targets": self._targets,
        }
        metrics: dict[str, int | float | str | bool] = {
            "k": self._config.k,
            "sample_count": self._features.shape[0],
            "class_count": len(set(self._targets.tolist())),
            "has_prediction": self._last_prediction is not None,
        }
        annotations = ("Classifier is fitted and ready for predictions.",)

        if self._last_prediction is not None:
            visual_state["query_point"] = self._last_prediction.query_point
            visual_state["neighbors"] = self._last_prediction.neighbors
            visual_state["vote_counts"] = self._last_prediction.vote_counts
            metrics["predicted_label"] = self._last_prediction.predicted_label
            annotations = (
                f"Predicted class {self._last_prediction.predicted_label}.",
                f"Used {self._config.k} nearest neighbors for voting.",
            )

        return AlgorithmSnapshot(
            iteration=0,
            status="ready",
            visual_state=visual_state,
            metrics=metrics,
            annotations=annotations,
            done=True,
        )

    def _ensure_fitted(self) -> None:
        """Ensure that the classifier has training data."""
        if not self.is_fitted:
            msg = "The classifier must be fitted before prediction."
            raise RuntimeError(msg)


def _extract_classification_arrays(dataset: DatasetSnapshot) -> tuple[FloatArray, IntArray]:
    """Extract and validate classification arrays from a dataset."""
    if dataset.targets is None:
        msg = "Dataset targets are required for classification."
        raise ValueError(msg)

    features = np.asarray(dataset.features, dtype=float)
    targets = np.asarray(dataset.targets, dtype=int)

    if features.ndim != EXPECTED_FEATURE_DIMENSIONS:
        msg = "Dataset features must be a two-dimensional array."
        raise ValueError(msg)

    if features.shape[0] == 0:
        msg = "Dataset cannot be empty."
        raise ValueError(msg)

    if features.shape[1] == 0:
        msg = "Dataset must contain at least one feature."
        raise ValueError(msg)

    if targets.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = "Dataset targets must be a one-dimensional array."
        raise ValueError(msg)

    if features.shape[0] != targets.shape[0]:
        msg = (
            "Dataset features and targets must contain the same number of samples. "
            f"Got {features.shape[0]} and {targets.shape[0]}."
        )
        raise ValueError(msg)

    return features, targets


def _count_votes(neighbors: tuple[Neighbor, ...]) -> dict[int, int]:
    """Count votes from nearest neighbors."""
    counter = Counter(neighbor.label for neighbor in neighbors)

    return dict(counter)


def _select_winning_label(vote_counts: dict[int, int]) -> int:
    """Select winning class label from vote counts.

    Ties are resolved by selecting the smallest class label. This deterministic
    rule makes tests and visualizations reproducible.
    """
    if not vote_counts:
        msg = "Cannot select a class without votes."
        raise ValueError(msg)

    max_votes = max(vote_counts.values())
    tied_labels = [label for label, count in vote_counts.items() if count == max_votes]

    return min(tied_labels)


def _validate_config(config: KNearestNeighborsConfig) -> None:
    """Validate k-NN configuration."""
    if config.k <= 0:
        msg = "k must be greater than 0."
        raise ValueError(msg)


def _validate_query_point(query_point: FloatArray, *, expected_feature_count: int) -> None:
    """Validate one query point against the expected feature count."""
    if query_point.ndim != EXPECTED_TARGET_DIMENSIONS:
        msg = "query_point must be a one-dimensional array."
        raise ValueError(msg)

    if query_point.size == 0:
        msg = "query_point cannot be empty."
        raise ValueError(msg)

    if query_point.shape[0] != expected_feature_count:
        msg = (
            "query_point must have the same number of features as training data. "
            f"Got {query_point.shape[0]} and {expected_feature_count}."
        )
        raise ValueError(msg)
