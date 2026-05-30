"""Unified app shell for Interactive ML Labs."""

from interactive_ml_labs.manifest import ControlBinding, DemoManifest, LocalizedText
from interactive_ml_labs.registry import (
    DEMO_MANIFESTS,
    LEVEL_NAMES,
    demos_for_level,
    levels_from_manifests,
)
from interactive_ml_labs.settings import AppContext, AppSettings

__all__ = [
    "DEMO_MANIFESTS",
    "LEVEL_NAMES",
    "AppContext",
    "AppSettings",
    "ControlBinding",
    "DemoManifest",
    "LocalizedText",
    "demos_for_level",
    "levels_from_manifests",
]
