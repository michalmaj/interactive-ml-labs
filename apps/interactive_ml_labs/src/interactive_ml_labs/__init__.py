"""Unified app shell for Interactive ML Labs."""

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
    "DEMO_BY_ID",
    "DEMO_MANIFESTS",
    "LEVEL_MANIFESTS",
    "LEVEL_NAMES",
    "POLISH_FONT_CANDIDATES",
    "POLISH_SAMPLE_TEXT",
    "AppContext",
    "AppSettings",
    "ControlBinding",
    "DemoManifest",
    "LevelManifest",
    "LocalizedText",
    "Scene",
    "SceneCommand",
    "SceneCommandKind",
    "SceneManager",
    "demos_for_level",
    "levels_from_manifests",
    "make_ui_font",
    "validate_demo_registry",
]
