"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

import pygame
from interactive_ml_labs.pygame_app import UnifiedAppShell
from interactive_ml_labs.settings import AppSettings


def test_shell_can_render_initial_screen(monkeypatch) -> None:
    """The shell should render its first screen without opening a real window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._render()
    finally:
        pygame.quit()
