"""Smoke tests for Decision Tree Pygame-related modules.

These tests only import modules. They do not initialize the graphical window.
"""

import decision_tree_splitter.pygame_app
import decision_tree_splitter.renderer
import decision_tree_splitter.scene


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert decision_tree_splitter.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert decision_tree_splitter.renderer is not None


def test_import_scene_module() -> None:
    """The reusable Pygame scene module should be importable."""

    assert decision_tree_splitter.scene is not None
