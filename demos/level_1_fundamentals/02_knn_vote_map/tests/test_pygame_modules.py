"""Smoke tests for k-NN Pygame-related modules.

These tests only import modules. They do not initialize the graphical window.
"""

import knn_vote_map.pygame_app
import knn_vote_map.renderer


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert knn_vote_map.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert knn_vote_map.renderer is not None
