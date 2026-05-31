"""Smoke tests for Pygame-related modules.

These tests only import modules. They do not initialize the graphical window.
"""

import random_forest_bagging_lab.pygame_app
import random_forest_bagging_lab.renderer
import random_forest_bagging_lab.scene


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert random_forest_bagging_lab.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert random_forest_bagging_lab.renderer is not None


def test_import_scene_module() -> None:
    """The reusable Pygame scene module should be importable."""

    assert random_forest_bagging_lab.scene is not None
