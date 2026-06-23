"""Tests for the unified app shell manifest registry."""

import pytest
from interactive_ml_labs import (
    DEMO_BY_ID,
    DEMO_MANIFESTS,
    LEARNING_PATH_MANIFESTS,
    LESSON_BY_ID,
    LESSON_MANIFESTS,
    LEVEL_MANIFESTS,
    LEVEL_NAMES,
    DemoManifest,
    LearningPathManifest,
    LessonManifest,
    LessonTask,
    LocalizedText,
    demos_for_level,
    levels_from_manifests,
    validate_demo_registry,
    validate_learning_registry,
)
from interactive_ml_labs.activation_scene import create_activation_functions_lab_scene
from interactive_ml_labs.anomaly_detection_scene import create_anomaly_detection_lab_scene
from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.calibration_scene import create_calibration_lab_scene
from interactive_ml_labs.class_imbalance_scene import create_class_imbalance_lab_scene
from interactive_ml_labs.clustering_scene import create_clustering_lab_scene
from interactive_ml_labs.data_leakage_scene import create_data_leakage_lab_scene
from interactive_ml_labs.decision_tree_scene import create_decision_tree_scene
from interactive_ml_labs.distance_metrics_scene import create_distance_metrics_lab_scene
from interactive_ml_labs.feature_importance_scene import create_feature_importance_lab_scene
from interactive_ml_labs.feature_scaling_scene import create_feature_scaling_lab_scene
from interactive_ml_labs.gaussian_mixture_scene import create_gaussian_mixture_intro_lab_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.kmeans_intro_scene import create_kmeans_intro_lab_scene
from interactive_ml_labs.knn_scene import create_knn_vote_map_scene
from interactive_ml_labs.linear_regression_scene import create_linear_regression_line_fit_lab_scene
from interactive_ml_labs.logistic_scene import create_logistic_regression_scene
from interactive_ml_labs.model_comparison_scene import create_model_comparison_lab_scene
from interactive_ml_labs.monitoring_scene import create_model_monitoring_drift_scene
from interactive_ml_labs.neural_network_scene import create_neural_network_playground_scene
from interactive_ml_labs.pca_scene import create_pca_lab_scene
from interactive_ml_labs.random_forest_scene import create_random_forest_scene
from interactive_ml_labs.split_lab_scene import create_train_validation_test_lab_scene
from interactive_ml_labs.svm_margin_scene import create_svm_margin_lab_scene
from interactive_ml_labs.time_series_scene import create_time_series_forecasting_lab_scene
from interactive_ml_labs.tsne_umap_scene import create_tsne_umap_exploration_scene
from interactive_ml_labs.tuning_scene import create_hyperparameter_tuning_lab_scene


def test_registry_contains_current_demo_levels() -> None:
    """Manifest registry should expose levels dynamically."""
    assert levels_from_manifests() == (1, 2, 3)
    assert 1 in LEVEL_NAMES
    assert 2 in LEVEL_NAMES
    assert 3 in LEVEL_NAMES
    assert {level.number for level in LEVEL_MANIFESTS} >= {1, 2, 3}


def test_registry_groups_demos_by_level() -> None:
    """Demo lookup should return only demos from the requested level."""
    level_one_demos = demos_for_level(1)
    level_two_demos = demos_for_level(2)
    level_three_demos = demos_for_level(3)

    assert {demo.level for demo in level_one_demos} == {1}
    assert {demo.level for demo in level_two_demos} == {2}
    assert {demo.level for demo in level_three_demos} == {3}
    assert len(level_one_demos) >= 10
    assert len(level_two_demos) >= 7
    assert len(level_three_demos) == 7
    assert level_three_demos[0].id == "clustering_lab"
    assert level_three_demos[1].id == "pca_lab"
    assert level_three_demos[2].id == "model_comparison_lab"
    assert level_three_demos[3].id == "calibration_lab"
    assert level_three_demos[4].id == "tsne_umap_exploration_lab"
    assert level_three_demos[5].id == "model_monitoring_drift_lab"
    assert level_three_demos[6].id == "time_series_forecasting_lab"


def test_linear_regression_manifest_describes_foundation_level_one_lab() -> None:
    """Linear Regression Line Fit Lab should extend the foundation track."""
    manifest = DEMO_BY_ID["linear_regression_line_fit_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_linear_regression_line_fit_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == ("regression", "loss", "residuals", "level-1")
    assert "Linear Regression Line Fit Lab" in text
    assert "linear regression" in text
    assert "residual" in text
    assert "MSE loss" in text
    assert "least squares" in text
    assert "Left / Right" in text
    assert "F" in text


def test_kmeans_intro_manifest_describes_foundation_level_one_lab() -> None:
    """K-Means Intro Lab should introduce unsupervised clustering basics."""
    manifest = DEMO_BY_ID["kmeans_intro_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_kmeans_intro_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == ("clustering", "k-means", "unsupervised", "inertia", "level-1")
    assert "K-Means Intro Lab" in text
    assert "K-Means" in text
    assert "centroid" in text
    assert "assignment" in text
    assert "inertia" in text
    assert "unsupervised" in text
    assert "Space" in text
    assert "- / =" in text
    assert "C" in text


def test_distance_metrics_manifest_describes_foundation_level_one_lab() -> None:
    """Distance Metrics Lab should bridge feature space intuition into k-NN."""
    manifest = DEMO_BY_ID["distance_metrics_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_distance_metrics_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == ("distance", "knn", "classification", "level-1")
    assert "Distance Metrics Lab" in text
    assert "Euclidean" in text
    assert "Manhattan" in text
    assert "Chebyshev" in text
    assert "nearest neighbor" in text
    assert "Arrow keys" in text
    assert "M" in text


def test_svm_margin_manifest_describes_foundation_level_one_lab() -> None:
    """SVM Margin Lab should extend the foundation classification track."""
    manifest = DEMO_BY_ID["svm_margin_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_svm_margin_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == ("classification", "svm", "margin", "boundary", "level-1")
    assert "SVM Margin Lab" in text
    assert "support vector" in text
    assert "margin" in text
    assert "decision boundary" in text
    assert "maximum margin" in text
    assert "Left / Right" in text
    assert "F" in text


def test_activation_manifest_describes_foundation_level_one_lab() -> None:
    """Activation Functions Lab should introduce neural-network building blocks."""
    manifest = DEMO_BY_ID["activation_functions_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_activation_functions_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == ("neural-networks", "activation", "gradient", "level-1")
    assert "Activation Functions Lab" in text
    assert "sigmoid" in text
    assert "tanh" in text
    assert "ReLU" in text
    assert "local gradient" in text
    assert "saturation" in text
    assert "Left / Right" in text


def test_neural_network_manifest_describes_foundation_level_one_lab() -> None:
    """Neural Network Playground should introduce a tiny forward pass."""
    manifest = DEMO_BY_ID["neural_network_playground"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 1
    assert manifest.create_scene is create_neural_network_playground_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Wprowadzający"
    assert manifest.tags == (
        "neural-networks",
        "forward-pass",
        "classification",
        "loss",
        "level-1",
    )
    assert "Neural Network Playground" in text
    assert "forward pass" in text
    assert "hidden layer" in text
    assert "weight" in text
    assert "bias" in text
    assert "loss" in text
    assert "- / =" in text


def test_data_leakage_manifest_describes_practical_level_two_lab() -> None:
    """Data Leakage Lab should extend the practical ML track."""
    manifest = DEMO_BY_ID["data_leakage_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_data_leakage_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("data-quality", "validation", "leakage", "level-2")
    assert "Data Leakage Lab" in text
    assert "data leakage" in text
    assert "prediction time" in text
    assert "target proxy" in text
    assert "validation" in text
    assert "L" in text


def test_class_imbalance_manifest_describes_practical_level_two_lab() -> None:
    """Class Imbalance Lab should extend the practical metrics track."""
    manifest = DEMO_BY_ID["class_imbalance_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_class_imbalance_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("classification", "metrics", "imbalance", "threshold", "level-2")
    assert "Class Imbalance Lab" in text
    assert "class imbalance" in text
    assert "precision" in text
    assert "recall" in text
    assert "false negative" in text
    assert "- / = / 0" in text


def test_train_validation_test_manifest_describes_practical_level_two_lab() -> None:
    """Train / Validation / Test Split Lab should extend the practical validation track."""
    manifest = DEMO_BY_ID["train_validation_test_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_train_validation_test_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("validation", "model-selection", "overfitting", "level-2")
    assert "Train / Validation / Test Split Lab" in text
    assert "train set" in text
    assert "validation set" in text
    assert "test set" in text
    assert "model selection" in text
    assert "- / = / 0" in text


def test_feature_scaling_manifest_describes_practical_level_two_lab() -> None:
    """Feature Scaling Lab should extend the practical preprocessing track."""
    manifest = DEMO_BY_ID["feature_scaling_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_feature_scaling_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("preprocessing", "scaling", "distance", "optimization", "level-2")
    assert "Feature Scaling Lab" in text
    assert "feature scaling" in text
    assert "standardization" in text
    assert "range ratio" in text
    assert "feature dominance" in text
    assert "S" in text


def test_feature_importance_manifest_describes_practical_level_two_lab() -> None:
    """Feature Importance Lab should teach careful importance interpretation."""
    manifest = DEMO_BY_ID["feature_importance_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_feature_importance_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == (
        "interpretability",
        "feature-importance",
        "leakage",
        "correlation",
        "level-2",
    )
    assert "Feature Importance Lab" in text
    assert "feature importance" in text
    assert "permutation importance" in text
    assert "model importance" in text
    assert "leakage" in text
    assert "correlated features" in text
    assert "causality" in text
    assert "M" in text
    assert "C" in text
    assert "L" in text


def test_gaussian_mixture_manifest_describes_practical_level_two_lab() -> None:
    """Gaussian Mixture Intro Lab should add probabilistic clustering intuition."""
    manifest = DEMO_BY_ID["gaussian_mixture_intro_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_gaussian_mixture_intro_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("clustering", "probability", "gmm", "soft-assignment", "level-2")
    assert "Gaussian Mixture Intro Lab" in text
    assert "Gaussian Mixture Model" in text
    assert "Gaussian component" in text
    assert "responsibility" in text
    assert "mixture weight" in text
    assert "covariance" in text
    assert "hard assignment" in text
    assert "Arrow keys" in text
    assert "- / =" in text


def test_anomaly_detection_manifest_describes_practical_level_two_lab() -> None:
    """Anomaly Detection Lab should extend practical alert threshold intuition."""
    manifest = DEMO_BY_ID["anomaly_detection_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_anomaly_detection_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("anomaly-detection", "threshold", "monitoring", "metrics", "level-2")
    assert "Anomaly Detection Lab" in text
    assert "anomaly score" in text
    assert "threshold" in text
    assert "false positive" in text
    assert "false negative" in text
    assert "alert" in text
    assert "- / = / 0" in text
    assert "S" in text


def test_hyperparameter_tuning_manifest_describes_practical_level_two_lab() -> None:
    """Hyperparameter Tuning Lab should extend the practical model-selection track."""
    manifest = DEMO_BY_ID["hyperparameter_tuning_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 2
    assert manifest.create_scene is create_hyperparameter_tuning_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Praktyczny"
    assert manifest.tags == ("validation", "tuning", "overfitting", "model-selection", "level-2")
    assert "Hyperparameter Tuning Lab" in text
    assert "hyperparameter" in text
    assert "validation curve" in text
    assert "grid search" in text
    assert "overfitting" in text
    assert "- / = / 0" in text


def test_time_series_forecasting_manifest_describes_level_three_lab() -> None:
    """Time Series Forecasting Lab should replace the Level 3 placeholder."""
    manifest = DEMO_BY_ID["time_series_forecasting_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.create_scene is create_time_series_forecasting_lab_scene
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Zaawansowany"
    assert manifest.tags == (
        "time-series",
        "forecasting",
        "uncertainty",
        "evaluation",
        "level-3",
    )
    assert "Time Series Forecasting Lab" in text
    assert "holdout window" in text
    assert "forecast horizon" in text
    assert "moving average" in text
    assert "trend-seasonal" in text
    assert "residual" in text
    assert "MAE" in text
    assert "RMSE" in text
    assert "forecast bias" in text
    assert "uncertainty" in text
    assert "1-3" in text
    assert "M" in text
    assert "- / =" in text
    assert "U" in text
    assert "E" in text
    assert "R" in text
    assert "T" in text


def test_model_monitoring_manifest_describes_level_three_lab() -> None:
    """Model Monitoring Drift Lab should describe the current Level 3 lab."""
    manifest = DEMO_BY_ID["model_monitoring_drift_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.en for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Prototyp zaawansowany"
    assert manifest.tags == ("monitoring", "drift", "production-ml", "evaluation", "level-3")
    assert manifest.create_scene is create_model_monitoring_drift_scene
    assert "Model Monitoring Drift Lab" in text
    assert "data drift" in text
    assert "metric drift" in text
    assert "monitoring window" in text
    assert "alert threshold" in text
    assert "alert rate" in text
    assert "persistence" in text
    assert "lead signal" in text
    assert "trend" in text
    assert "press A" in text
    assert "single spike" in text
    assert "persistent drift" in text
    assert "baseline" in text
    assert "D / M" in text
    assert "- / =" in text
    assert "R" in text
    assert "T" in text


def test_clustering_manifest_sets_first_level_three_demo_contract() -> None:
    """Clustering Lab should be the first real Level 3 demo contract."""
    manifest = DEMO_BY_ID["clustering_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Zaawansowane"
    assert manifest.tags == ("clustering", "k-means", "dbscan", "unsupervised", "visualization")
    assert manifest.create_scene is create_clustering_lab_scene
    assert "K-Means" in text
    assert "centroid" in text
    assert "inertia" in text
    assert "Lesson path" in text
    assert "Dragging" in text
    assert "point-to-centroid" in text
    assert "DBSCAN" in text
    assert "eps" in text
    assert "noise" in text
    assert "What to compare" in text
    assert "Space" in text
    assert "C" in text
    assert "M" in text
    assert "Mouse drag" in text


def test_pca_manifest_sets_second_level_three_demo_contract() -> None:
    """PCA Lab should define the first dimensionality reduction demo contract."""
    manifest = DEMO_BY_ID["pca_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Zaawansowany podgląd"
    assert manifest.tags == (
        "pca",
        "dimensionality-reduction",
        "projection",
        "visualization",
    )
    assert manifest.create_scene is create_pca_lab_scene
    assert "PCA" in text
    assert "projection" in text
    assert "explained variance" in text
    assert "principal component" in text
    assert "Lesson path" in text
    assert "Data and noise" in text
    assert "Residuals and reconstruction" in text
    assert "What to compare" in text
    assert "1-3" in text
    assert "- / =" in text
    assert "N" in text
    assert "Left / Right" in text
    assert "F" in text
    assert "C" in text
    assert "residual" in text
    assert "reconstruction error" in text
    assert "R" in text
    assert "T" in text


def test_model_comparison_manifest_sets_third_level_three_demo_contract() -> None:
    """Model Comparison Lab should define the first classifier comparison contract."""
    manifest = DEMO_BY_ID["model_comparison_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Zaawansowany podgląd"
    assert manifest.tags == (
        "model-comparison",
        "classification",
        "decision-boundary",
        "visualization",
    )
    assert manifest.create_scene is create_model_comparison_lab_scene
    assert "Model Comparison Lab" in text
    assert "Logistic Regression" in text
    assert "k-NN" in text
    assert "Decision Tree" in text
    assert "decision boundary" in text
    assert "model assumption" in text
    assert "train/test accuracy" in text
    assert "confusion" in text
    assert "precision/recall" in text
    assert "misclassified test points" in text
    assert "Lesson path" in text
    assert "Same data" in text
    assert "dataset presets" in text
    assert "1" in text
    assert "2" in text
    assert "3" in text
    assert "- / =" in text
    assert "A" in text
    assert "D" in text
    assert "E" in text
    assert "T" in text


def test_calibration_manifest_sets_fourth_level_three_demo_contract() -> None:
    """Calibration Lab should define the first probability calibration contract."""
    manifest = DEMO_BY_ID["calibration_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Zaawansowany podgląd"
    assert manifest.tags == ("calibration", "probability", "evaluation", "visualization")
    assert manifest.create_scene is create_calibration_lab_scene
    assert "Calibration Lab" in text
    assert "calibration" in text
    assert "reliability diagram" in text
    assert "Brier score" in text
    assert "ECE" in text
    assert "confidence" in text
    assert "observed frequenc" in text
    assert "1-3" in text
    assert "- / =" in text
    assert "temperature scaling" in text
    assert "accuracy@0.5" in text
    assert "threshold" in text
    assert "worst gap" in text
    assert "O" in text
    assert "raw" in text
    assert "legend" in text
    assert "E" in text
    assert "R" in text
    assert "T" in text


def test_tsne_umap_manifest_sets_fifth_level_three_demo_contract() -> None:
    """t-SNE / UMAP Exploration Lab should define the Level 3 prototype contract."""
    manifest = DEMO_BY_ID["tsne_umap_exploration_lab"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.key for control in manifest.controls),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
            *(section.title.en for section in manifest.theory.sections),
            *(paragraph.en for section in manifest.theory.sections for paragraph in section.body),
            *(entry.term for entry in manifest.theory.glossary),
        ],
    )

    assert manifest.level == 3
    assert manifest.difficulty is not None
    assert manifest.difficulty.pl == "Prototyp zaawansowany"
    assert manifest.tags == (
        "tsne",
        "umap",
        "embedding",
        "dimensionality-reduction",
        "visualization",
    )
    assert manifest.create_scene is create_tsne_umap_exploration_scene
    assert "t-SNE / UMAP Exploration Lab" in text
    assert "Explore an interactive embedding prototype" in text
    assert "exploration lesson" in text
    assert "embedding" in text
    assert "perplexity" in text
    assert "neighbors" in text
    assert "seed" in text
    assert "seed variant" in text
    assert "local neighborhoods" in text
    assert "Local trust" in text
    assert "Global spread" in text
    assert "seed drift" in text
    assert "color legend" in text
    assert "Dataset cues" in text
    assert "global structure" in text
    assert "1-3" in text
    assert "M" in text
    assert "- / =" in text
    assert "S" in text
    assert "L" in text
    assert "O" in text
    assert "raw" in text
    assert "high-dimensional" in text
    assert "R" in text
    assert "T" in text


def test_manifests_have_required_teaching_content() -> None:
    """Every manifest should include content used by generated intro screens."""
    for manifest in DEMO_MANIFESTS:
        assert manifest.id
        assert manifest.title.en
        assert manifest.title.pl
        assert manifest.summary.en
        assert manifest.summary.pl
        assert manifest.objectives
        assert manifest.controls
        assert manifest.tags
        assert manifest.create_scene is not None


def test_boosting_manifest_has_demo_specific_teaching_content() -> None:
    """Boosting manifest should describe the real demo, not only placeholder content."""
    manifest = DEMO_BY_ID["boosting_mistake_lab"]
    polish_text = " ".join(
        [
            manifest.summary.pl,
            *(objective.pl for objective in manifest.objectives),
            *(control.action.pl for control in manifest.controls),
        ],
    )

    assert "weak learners" in polish_text
    assert "train/test accuracy" in polish_text
    assert "generalization gap" in polish_text
    assert "confidence view" in polish_text
    assert "decision boundary" in polish_text
    assert len(manifest.objectives) == 3
    assert len(manifest.controls) >= 5
    assert manifest.create_scene is create_boosting_mistake_lab_scene


def test_gradient_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Gradient manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["gradient_descent_playground"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_gradient_descent_scene
    assert "gradient descent" in text
    assert "learning rate" in text
    assert "dataset noise" in text
    assert len(manifest.controls) >= 6


def test_manifests_have_in_app_theory_content() -> None:
    """Every integrated demo should include complete lesson content for the theory screen."""
    for manifest in DEMO_MANIFESTS:
        assert manifest.theory is not None
        assert len(manifest.theory.sections) >= 3
        assert len(manifest.theory.mini_challenges) >= 2
        assert len(manifest.theory.glossary) >= 2

        for section in manifest.theory.sections:
            assert section.title.en
            assert section.title.pl
            assert section.body
            assert all(paragraph.en and paragraph.pl for paragraph in section.body)

        assert all(challenge.en and challenge.pl for challenge in manifest.theory.mini_challenges)
        for entry in manifest.theory.glossary:
            assert entry.term
            assert entry.definition.en
            assert entry.definition.pl


@pytest.mark.parametrize(
    ("demo_id", "expected_terms"),
    [
        ("gradient_descent_playground", ("Gradient descent", "learning rate", "loss")),
        ("knn_vote_map", ("k-NN", "query", "vote map")),
        ("logistic_regression_boundary_lab", ("Logistic regression", "threshold", "precision")),
        ("decision_tree_splitter", ("Decision tree", "manual split", "impurity")),
        ("random_forest_bagging_lab", ("Random forest", "bootstrap", "confidence view")),
        ("boosting_mistake_lab", ("Boosting", "weak learner", "staged train/test accuracy")),
        (
            "model_comparison_lab",
            ("model comparison", "decision boundary", "model assumption"),
        ),
        (
            "calibration_lab",
            (
                "calibration",
                "reliability diagram",
                "Brier score",
                "ECE",
                "temperature scaling",
                "accuracy@0.5",
                "worst gap",
            ),
        ),
        (
            "tsne_umap_exploration_lab",
            (
                "t-SNE",
                "UMAP",
                "embedding",
                "perplexity",
                "neighbors",
                "local trust",
                "global spread",
                "seed drift",
                "raw layout",
            ),
        ),
    ],
)
def test_demo_theory_contains_key_terms(demo_id: str, expected_terms: tuple[str, ...]) -> None:
    """Theory copy should preserve important domain vocabulary."""
    manifest = DEMO_BY_ID[demo_id]

    theory_text = " ".join(
        [
            *(section.title.pl for section in manifest.theory.sections),
            *(paragraph.pl for section in manifest.theory.sections for paragraph in section.body),
            *(challenge.pl for challenge in manifest.theory.mini_challenges),
            *(entry.term for entry in manifest.theory.glossary),
            *(entry.definition.pl for entry in manifest.theory.glossary),
        ],
    ).lower()

    for term in expected_terms:
        assert term.lower() in theory_text


def test_knn_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """k-NN manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["knn_vote_map"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_knn_vote_map_scene
    assert "k-NN" in text
    assert "query point" in text
    assert "dataset noise" in text
    assert len(manifest.controls) >= 6


def test_logistic_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Logistic Regression manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["logistic_regression_boundary_lab"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_logistic_regression_scene
    assert "logistic regression" in text
    assert "learning rate" in text
    assert "threshold" in text
    assert "precision" in text
    assert "recall" in text
    assert len(manifest.controls) >= 7


def test_decision_tree_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Decision Tree manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["decision_tree_splitter"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_decision_tree_scene
    assert "tree splits" in text
    assert "manual split" in text
    assert "max depth" in text
    assert "Gini" in text
    assert "entropy" in text
    assert len(manifest.controls) >= 9


def test_random_forest_manifest_has_real_scene_and_demo_specific_controls() -> None:
    """Random Forest manifest should route to the real scene and describe its controls."""
    manifest = DEMO_BY_ID["random_forest_bagging_lab"]
    text = " ".join(
        [
            manifest.summary.en,
            *(objective.en for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
        ],
    )

    assert manifest.create_scene is create_random_forest_scene
    assert "random forest" in text
    assert "single tree" in text
    assert "bootstrap" in text
    assert "confidence view" in text
    assert len(manifest.controls) >= 8


def test_registry_contains_polish_diacritics() -> None:
    """Polish registry text should preserve diacritics."""
    polish_text = " ".join(
        [
            *(manifest.title.pl for manifest in DEMO_MANIFESTS),
            *(manifest.summary.pl for manifest in DEMO_MANIFESTS),
        ],
    )

    assert "ł" in polish_text
    assert "ó" in polish_text
    assert "ę" in polish_text


def test_registry_validates_default_manifests() -> None:
    """Default registry should pass consistency validation."""
    validate_demo_registry()
    validate_learning_registry()


def test_learning_path_registry_contains_models_learn_from_error_path() -> None:
    """Learning paths should connect existing demos into a guided story."""
    path = LEARNING_PATH_MANIFESTS[0]

    assert path.id == "models_learn_from_error"
    assert path.title.en == "How models learn from error"
    assert path.title.pl == "Jak modele uczą się z błędu"
    assert path.lesson_ids == (
        "error_linear_regression_line_fit",
        "error_gradient_descent",
        "error_logistic_boundary",
        "error_boosting_mistakes",
    )

    demos_in_path = [LESSON_BY_ID[lesson_id].demo_id for lesson_id in path.lesson_ids]
    assert demos_in_path == [
        "linear_regression_line_fit_lab",
        "gradient_descent_playground",
        "logistic_regression_boundary_lab",
        "boosting_mistake_lab",
    ]
    assert all(demo_id in DEMO_BY_ID for demo_id in demos_in_path)


def test_learning_lessons_define_goals_tasks_and_badges() -> None:
    """Every default lesson should have enough metadata for a future lesson screen."""
    assert len(LESSON_MANIFESTS) == 4

    for lesson in LESSON_MANIFESTS:
        assert lesson.title.en
        assert lesson.title.pl
        assert lesson.learning_goal.en
        assert lesson.learning_goal.pl
        assert lesson.completion_badge is not None
        assert lesson.completion_badge.en
        assert lesson.completion_badge.pl
        assert len(lesson.tasks) >= 2
        assert all(task.title.en and task.title.pl for task in lesson.tasks)
        assert all(task.instruction.en and task.instruction.pl for task in lesson.tasks)
        assert all(task.success_condition for task in lesson.tasks)

    assert LESSON_BY_ID["error_gradient_descent"].prerequisites == (
        "error_linear_regression_line_fit",
    )
    assert LESSON_BY_ID["error_boosting_mistakes"].level == 2


def test_registry_rejects_duplicate_demo_ids() -> None:
    """Registry validation should reject duplicate demo identifiers."""
    duplicate = DEMO_MANIFESTS[0]

    with pytest.raises(ValueError, match="Duplicate demo ids"):
        validate_demo_registry(demo_manifests=(duplicate, duplicate))


def test_registry_rejects_unknown_demo_level() -> None:
    """Registry validation should reject demos assigned to unknown levels."""
    invalid_demo = DemoManifest(
        id="unknown_level_demo",
        level=99,
        title=LocalizedText(en="Unknown", pl="Nieznany"),
        summary=LocalizedText(en="Unknown level", pl="Nieznany poziom"),
        objectives=DEMO_MANIFESTS[0].objectives,
        controls=DEMO_MANIFESTS[0].controls,
        create_scene=DEMO_MANIFESTS[0].create_scene,
        tags=("test",),
    )

    with pytest.raises(ValueError, match="unknown level"):
        validate_demo_registry(demo_manifests=(invalid_demo,))


def test_learning_registry_rejects_unknown_demo_reference() -> None:
    """Learning validation should reject lessons for demos that do not exist."""
    invalid_lesson = LessonManifest(
        id="missing_demo_lesson",
        level=1,
        demo_id="missing_demo",
        title=LocalizedText(en="Missing", pl="Brak"),
        learning_goal=LocalizedText(en="Learn something", pl="Naucz się czegoś"),
        tasks=(
            LessonTask(
                id="task",
                title=LocalizedText(en="Task", pl="Zadanie"),
                instruction=LocalizedText(en="Do it", pl="Zrób to"),
                success_condition="started_demo",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown demo"):
        validate_learning_registry(lesson_manifests=(invalid_lesson,))


def test_learning_registry_rejects_unknown_lesson_in_path() -> None:
    """Learning validation should reject paths that reference missing lessons."""
    invalid_path = LearningPathManifest(
        id="broken_path",
        title=LocalizedText(en="Broken", pl="Zepsuta"),
        summary=LocalizedText(en="Broken path", pl="Zepsuta ścieżka"),
        lesson_ids=("missing_lesson",),
    )

    with pytest.raises(ValueError, match="unknown lesson"):
        validate_learning_registry(learning_path_manifests=(invalid_path,))
