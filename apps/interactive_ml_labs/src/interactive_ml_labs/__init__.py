"""Unified app shell for Interactive ML Labs."""

from interactive_ml_labs.display import (
    BOOSTING_FIXED_SCENE_SIZE,
    DEFAULT_RESOLUTION,
    SCREEN_MARGIN,
    WINDOWED_RESOLUTIONS,
    RectTuple,
    Size,
    center_rect,
    choose_adaptive_window_size,
)
from interactive_ml_labs.fonts import POLISH_FONT_CANDIDATES, POLISH_SAMPLE_TEXT, make_ui_font
from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LevelManifest, LocalizedText
from interactive_ml_labs.registry import (
    DEMO_BY_ID,
    DEMO_MANIFESTS,
    LEVEL_MANIFESTS,
    LEVEL_NAMES,
    demos_for_level,
    levels_from_manifests,
    validate_demo_registry,
)
from interactive_ml_labs.scene import Scene, SceneCommand, SceneCommandKind, SceneManager
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
    "ControlBinding",
    "DemoManifest",
    "LevelManifest",
    "LocalizedText",
    "RectTuple",
    "Scene",
    "SceneCommand",
    "SceneCommandKind",
    "SceneManager",
    "Size",
    "center_rect",
    "choose_adaptive_window_size",
    "demos_for_level",
    "levels_from_manifests",
    "make_ui_font",
    "validate_demo_registry",
]
