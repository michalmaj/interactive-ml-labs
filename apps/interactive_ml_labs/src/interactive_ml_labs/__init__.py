"""Unified app shell for Interactive ML Labs."""

from interactive_ml_labs.boosting_scene import (
    BoostingMistakeLabSceneAdapter,
    create_boosting_mistake_lab_scene,
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
from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LevelManifest, LocalizedText
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
from interactive_ml_labs.settings import AppContext, AppSettings

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
    "AppContext",
    "AppSettings",
    "BoostingMistakeLabSceneAdapter",
    "ControlBinding",
    "DecisionTreeSceneAdapter",
    "DemoManifest",
    "FixedSizeScene",
    "GradientDescentSceneAdapter",
    "KNNVoteMapSceneAdapter",
    "LevelManifest",
    "LocalizedText",
    "LogisticRegressionSceneAdapter",
    "RandomForestSceneAdapter",
    "RectTuple",
    "Scene",
    "SceneCommand",
    "SceneCommandKind",
    "SceneManager",
    "Size",
    "center_rect",
    "choose_adaptive_window_size",
    "create_boosting_mistake_lab_scene",
    "create_decision_tree_scene",
    "create_gradient_descent_scene",
    "create_knn_vote_map_scene",
    "create_logistic_regression_scene",
    "create_random_forest_scene",
    "demos_for_level",
    "levels_from_manifests",
    "make_ui_font",
    "scale_rect_to_fit",
    "validate_demo_registry",
]
