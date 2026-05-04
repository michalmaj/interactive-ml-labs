"""k-NN Vote Map demo package."""

from knn_vote_map.classifier import (
    KNearestNeighborsClassifier,
    KNearestNeighborsConfig,
    Neighbor,
    PredictionResult,
)
from knn_vote_map.dataset import (
    SyntheticClassificationConfig,
    make_synthetic_classification_dataset,
)
from knn_vote_map.metrics import euclidean_distance, euclidean_distances

__all__ = [
    "KNearestNeighborsClassifier",
    "KNearestNeighborsConfig",
    "Neighbor",
    "PredictionResult",
    "SyntheticClassificationConfig",
    "euclidean_distance",
    "euclidean_distances",
    "make_synthetic_classification_dataset",
]
