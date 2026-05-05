"""Logistic Regression Boundary Lab demo package."""

from logistic_regression_boundary_lab.algorithm import (
    LogisticRegressionConfig,
    StepwiseLogisticRegression,
)
from logistic_regression_boundary_lab.dataset import (
    SyntheticBinaryClassificationConfig,
    make_synthetic_binary_classification_dataset,
)
from logistic_regression_boundary_lab.metrics import (
    ClassificationMetrics,
    ConfusionMatrixCounts,
    accuracy_score,
    binary_cross_entropy,
    classification_metrics,
    confusion_matrix_counts,
    precision_score,
    predict_labels_from_probabilities,
    recall_score,
    sigmoid,
)
from logistic_regression_boundary_lab.probability_grid import (
    ProbabilityGrid,
    compute_probability_grid,
)

__all__ = [
    "ClassificationMetrics",
    "ConfusionMatrixCounts",
    "LogisticRegressionConfig",
    "ProbabilityGrid",
    "StepwiseLogisticRegression",
    "SyntheticBinaryClassificationConfig",
    "accuracy_score",
    "binary_cross_entropy",
    "classification_metrics",
    "compute_probability_grid",
    "confusion_matrix_counts",
    "make_synthetic_binary_classification_dataset",
    "precision_score",
    "predict_labels_from_probabilities",
    "recall_score",
    "sigmoid",
]
