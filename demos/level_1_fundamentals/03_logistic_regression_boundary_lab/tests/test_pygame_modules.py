"""Smoke tests for Logistic Regression Pygame-related modules.

These tests only import modules. They do not initialize the graphical window.
"""

import logistic_regression_boundary_lab.pygame_app
import logistic_regression_boundary_lab.renderer


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert logistic_regression_boundary_lab.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert logistic_regression_boundary_lab.renderer is not None
