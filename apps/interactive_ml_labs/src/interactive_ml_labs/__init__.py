"""Unified app shell for Interactive ML Labs."""

from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LevelManifest, LocalizedText
from interactive_ml_labs.registry import (
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
    "DEMO_MANIFESTS",
    "LEVEL_MANIFESTS",
    "LEVEL_NAMES",
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
    "validate_demo_registry",
]
