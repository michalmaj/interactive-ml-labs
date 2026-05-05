"""k-NN Vote Map demo package."""

from knn_vote_map.challenge import (
    KNNAccuracyChallenge,
    KNNAccuracyChallengeConfig,
    KNNAccuracyChallengeResult,
    accuracy_score,
)
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
from knn_vote_map.decision_grid import DecisionGrid, compute_decision_grid
from knn_vote_map.explanation import build_explanation_lines
from knn_vote_map.metrics import euclidean_distance, euclidean_distances

__all__ = [
    "DecisionGrid",
    "KNNAccuracyChallenge",
    "KNNAccuracyChallengeConfig",
    "KNNAccuracyChallengeResult",
    "KNearestNeighborsClassifier",
    "KNearestNeighborsConfig",
    "Neighbor",
    "PredictionResult",
    "SyntheticClassificationConfig",
    "accuracy_score",
    "build_explanation_lines",
    "compute_decision_grid",
    "euclidean_distance",
    "euclidean_distances",
    "make_synthetic_classification_dataset",
]
