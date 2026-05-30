"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from interactive_ml_labs.pygame_app import ScreenName, UnifiedAppShell
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppSettings


class FixedSizeColorScene:
    """Tiny fixed-size scene double for shell scaling tests."""

    fixed_scene_size = (100, 50)

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        _ = event
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Fill the target surface with a visible color."""
        surface.fill((240, 20, 20))


def test_shell_can_render_initial_screen(monkeypatch) -> None:
    """The shell should render its first screen without opening a real window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._render()
    finally:
        pygame.quit()


def test_shell_applies_adaptive_window_size_when_enabled(monkeypatch) -> None:
    """The shell should resolve opt-in adaptive sizing before opening the window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(
        pygame.display,
        "Info",
        lambda: SimpleNamespace(current_w=1512, current_h=982),
    )

    app = UnifiedAppShell(settings=AppSettings(adaptive_window_enabled=True))

    try:
        assert app.context.settings.resolution == (1320, 780)
    finally:
        pygame.quit()


def test_shell_scales_fixed_size_scene_when_enabled(monkeypatch) -> None:
    """The shell should letterbox fixed-size scenes into the current window."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(
        settings=AppSettings(
            resolution=(200, 200),
            fixed_scene_scaling_enabled=True,
        ),
    )

    try:
        app._render_scene(FixedSizeColorScene())

        assert app.screen.get_at((100, 100))[:3] == (240, 20, 20)
        assert app.screen.get_at((100, 40))[:3] != (240, 20, 20)
    finally:
        pygame.quit()


def test_shell_hover_selects_menu_item(monkeypatch) -> None:
    """Mouse hover should move selection to the hovered menu item."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._render()
        second_item = app.menu_items[1]
        app._handle_mouse_motion(second_item.rect.center)

        assert app.selected_index == 1
    finally:
        pygame.quit()


def test_shell_click_outside_menu_does_not_activate_selection(monkeypatch) -> None:
    """Clicking outside menu items should not activate the current selection."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._render()
        app._handle_mouse_click((1, 1))

        assert app.screen_name == ScreenName.LANGUAGE
    finally:
        pygame.quit()


def test_shell_l_key_toggles_language(monkeypatch) -> None:
    """L should toggle language globally."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._handle_keydown(pygame.K_l)
        assert app.context.settings.language == "pl"

        app._handle_keydown(pygame.K_l)
        assert app.context.settings.language == "en"
    finally:
        pygame.quit()


def test_shell_backspace_goes_back_like_escape(monkeypatch) -> None:
    """Backspace should behave as a back navigation shortcut."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.LEVELS
        app._handle_keydown(pygame.K_BACKSPACE)

        assert app.screen_name == ScreenName.LANGUAGE
    finally:
        pygame.quit()
