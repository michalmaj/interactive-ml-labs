"""Unified app shell for Interactive ML Labs."""

from interactive_ml_labs.boosting_scene import (
    BoostingMistakeLabSceneAdapter,
    create_boosting_mistake_lab_scene,
)
from interactive_ml_labs.calibration_scene import CalibrationLabScene, create_calibration_lab_scene
from interactive_ml_labs.clustering_scene import (
    AlgorithmMode,
    ClusteringLabScene,
    create_clustering_lab_scene,
)
from interactive_ml_labs.decision_tree_scene import (
    DecisionTreeSceneAdapter,
    create_decision_tree_scene,
)
from interactive_ml_labs.display import (
    BOOSTING_FIXED_SCENE_SIZE,
    DEFAULT_RESOLUTION,
    SCREEN_MARGIN,
    WINDOWED_RESOLUTIONS,
    RectTuple,
    Size,
    center_rect,
    choose_adaptive_window_size,
    scale_rect_to_fit,
)
from interactive_ml_labs.fonts import POLISH_FONT_CANDIDATES, POLISH_SAMPLE_TEXT, make_ui_font
from interactive_ml_labs.gradient_scene import (
    GradientDescentSceneAdapter,
    create_gradient_descent_scene,
)
from interactive_ml_labs.knn_scene import KNNVoteMapSceneAdapter, create_knn_vote_map_scene
from interactive_ml_labs.logistic_scene import (
    LogisticRegressionSceneAdapter,
    create_logistic_regression_scene,
)
from interactive_ml_labs.manifest import (
    ControlBinding,
    DemoManifest,
    DemoTheory,
    LevelManifest,
    LocalizedText,
    TheorySection,
)
from interactive_ml_labs.model_comparison_scene import (
    ModelComparisonLabScene,
    create_model_comparison_lab_scene,
)
from interactive_ml_labs.pca_scene import PCALabScene, create_pca_lab_scene
from interactive_ml_labs.random_forest_scene import (
    RandomForestSceneAdapter,
    create_random_forest_scene,
)
from interactive_ml_labs.registry import (
    DEMO_BY_ID,
    DEMO_MANIFESTS,
    LEVEL_MANIFESTS,
    LEVEL_NAMES,
    demos_for_level,
    levels_from_manifests,
    validate_demo_registry,
)
from interactive_ml_labs.scene import (
    FixedSizeScene,
    Scene,
    SceneCommand,
    SceneCommandKind,
    SceneManager,
)
from interactive_ml_labs.settings import (
    AppContext,
    AppSettings,
    default_settings_path,
    load_app_settings,
    save_app_settings,
    settings_from_json,
    settings_to_json,
)

__all__ = [
    "BOOSTING_FIXED_SCENE_SIZE",
    "DEFAULT_RESOLUTION",
    "DEMO_BY_ID",
    "DEMO_MANIFESTS",
    "LEVEL_MANIFESTS",
    "LEVEL_NAMES",
    "POLISH_FONT_CANDIDATES",
    "POLISH_SAMPLE_TEXT",
    "SCREEN_MARGIN",
    "WINDOWED_RESOLUTIONS",
    "AlgorithmMode",
    "AppContext",
    "AppSettings",
    "BoostingMistakeLabSceneAdapter",
    "CalibrationLabScene",
    "ClusteringLabScene",
    "ControlBinding",
    "DecisionTreeSceneAdapter",
    "DemoManifest",
    "DemoTheory",
    "FixedSizeScene",
    "GradientDescentSceneAdapter",
    "KNNVoteMapSceneAdapter",
    "LevelManifest",
    "LocalizedText",
    "LogisticRegressionSceneAdapter",
    "ModelComparisonLabScene",
    "PCALabScene",
    "RandomForestSceneAdapter",
    "RectTuple",
    "Scene",
    "SceneCommand",
    "SceneCommandKind",
    "SceneManager",
    "Size",
    "TheorySection",
    "center_rect",
    "choose_adaptive_window_size",
    "create_boosting_mistake_lab_scene",
    "create_calibration_lab_scene",
    "create_clustering_lab_scene",
    "create_decision_tree_scene",
    "create_gradient_descent_scene",
    "create_knn_vote_map_scene",
    "create_logistic_regression_scene",
    "create_model_comparison_lab_scene",
    "create_pca_lab_scene",
    "create_random_forest_scene",
    "default_settings_path",
    "demos_for_level",
    "levels_from_manifests",
    "load_app_settings",
    "make_ui_font",
    "save_app_settings",
    "scale_rect_to_fit",
    "settings_from_json",
    "settings_to_json",
    "validate_demo_registry",
]
