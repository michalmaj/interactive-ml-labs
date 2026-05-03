"""Smoke tests for Pygame-related modules.

These tests only import modules. They do not initialize the graphical window.
"""

import gradient_descent_playground.pygame_app
import gradient_descent_playground.renderer


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert gradient_descent_playground.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert gradient_descent_playground.renderer is not None
