"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from interactive_ml_labs import DEMO_BY_ID
from interactive_ml_labs.boosting_scene import BoostingMistakeLabSceneAdapter
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


def test_shell_starts_boosting_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real Boosting scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["boosting_mistake_lab"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, BoostingMistakeLabSceneAdapter)
        assert app.screen_name == ScreenName.DEMO
    finally:
        pygame.quit()


def test_shell_help_overlay_uses_selected_demo_manifest(monkeypatch) -> None:
    """Help overlay should use manifest text for the selected demo."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    wrapped_text: list[str] = []

    def capture_wrapped(
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        _ = width, font, color
        wrapped_text.append(text)
        return position[1] + 24

    try:
        app.selected_demo = DEMO_BY_ID["boosting_mistake_lab"]
        app.screen_name = ScreenName.INTRO
        app.context.settings.language = "pl"
        app._draw_wrapped = capture_wrapped

        app._render_help_overlay()

        help_text = " ".join(wrapped_text)
        assert "weak learners" in help_text
        assert "generalization gap" in help_text
        assert "confidence view" in help_text
        assert "decision boundary" in help_text
    finally:
        pygame.quit()


def test_shell_pause_help_menu_toggles_visible_overlay(monkeypatch) -> None:
    """Pause menu Help should make the shared help overlay visible."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.PAUSE
        app.selected_index = 1
        app._activate_selected()

        assert app.help_visible is True
    finally:
        pygame.quit()


def test_shell_s_key_opens_settings_outside_demo(monkeypatch) -> None:
    """S should open settings from shell navigation screens."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.LEVELS
        app._handle_keydown(pygame.K_s)

        assert app.screen_name == ScreenName.SETTINGS
        assert app.settings_return_screen == ScreenName.LEVELS
    finally:
        pygame.quit()


def test_shell_settings_menu_toggles_display_flags(monkeypatch) -> None:
    """Settings menu should mutate in-memory display options."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.SETTINGS
        app.selected_index = 1
        app._activate_selected()
        assert app.context.settings.adaptive_window_enabled is True

        app.selected_index = 2
        app._activate_selected()
        assert app.context.settings.fixed_scene_scaling_enabled is False
    finally:
        pygame.quit()


def test_shell_settings_back_returns_to_previous_screen(monkeypatch) -> None:
    """Settings Back should return to the screen that opened settings."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.DEMOS
        app._open_settings()
        app.selected_index = 4
        app._activate_selected()

        assert app.screen_name == ScreenName.DEMOS
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
