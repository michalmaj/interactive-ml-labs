"""Tests for the unified app shell manifest registry."""

import pytest
from interactive_ml_labs import (
    DEMO_BY_ID,
    DEMO_MANIFESTS,
    LEVEL_MANIFESTS,
    LEVEL_NAMES,
    DemoManifest,
    LocalizedText,
    demos_for_level,
    levels_from_manifests,
    validate_demo_registry,
)
from interactive_ml_labs.boosting_scene import create_boosting_mistake_lab_scene
from interactive_ml_labs.calibration_scene import create_calibration_lab_scene
from interactive_ml_labs.class_imbalance_scene import create_class_imbalance_lab_scene
from interactive_ml_labs.clustering_scene import create_clustering_lab_scene
from interactive_ml_labs.data_leakage_scene import create_data_leakage_lab_scene
from interactive_ml_labs.decision_tree_scene import create_decision_tree_scene
from interactive_ml_labs.feature_scaling_scene import create_feature_scaling_lab_scene
from interactive_ml_labs.gradient_scene import create_gradient_descent_scene
from interactive_ml_labs.knn_scene import create_knn_vote_map_scene
from interactive_ml_labs.logistic_scene import create_logistic_regression_scene
from interactive_ml_labs.model_comparison_scene import create_model_comparison_lab_scene
from interactive_ml_labs.monitoring_scene import create_model_monitoring_drift_scene
from interactive_ml_labs.pca_scene import create_pca_lab_scene
from interactive_ml_labs.random_forest_scene import create_random_forest_scene
from interactive_ml_labs.split_lab_scene import create_train_validation_test_lab_scene
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
    assert len(level_one_demos) >= 4
    assert len(level_two_demos) >= 7
    assert len(level_three_demos) == 7
    assert level_three_demos[0].id == "clustering_lab"
    assert level_three_demos[1].id == "pca_lab"
    assert level_three_demos[2].id == "model_comparison_lab"
    assert level_three_demos[3].id == "calibration_lab"
    assert level_three_demos[4].id == "tsne_umap_exploration_lab"
    assert level_three_demos[5].id == "model_monitoring_drift_lab"
    assert level_three_demos[6].id == "level_3_coming_soon"


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


def test_level_three_placeholder_describes_future_advanced_demos() -> None:
    """Level 3 placeholder should explain the future advanced/showcase track."""
    manifest = DEMO_BY_ID["level_3_coming_soon"]
    text = " ".join(
        [
            manifest.title.en,
            manifest.title.pl,
            manifest.summary.en,
            manifest.summary.pl,
            *(objective.en for objective in manifest.objectives),
            *(objective.pl for objective in manifest.objectives),
            *(control.action.en for control in manifest.controls),
            *(control.action.pl for control in manifest.controls),
        ],
    )

    assert manifest.level == 3
    assert "Coming Soon" in text
    assert "advanced" in text
    assert "Level 3" in text
    assert "Coming soon" in text
    assert manifest.difficulty.pl == "W przygotowaniu"
    assert manifest.tags == ("level-3", "showcase", "planning")


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
