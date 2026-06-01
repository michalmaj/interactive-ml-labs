"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from interactive_ml_labs import DEMO_BY_ID
from interactive_ml_labs.boosting_scene import BoostingMistakeLabSceneAdapter
from interactive_ml_labs.decision_tree_scene import DecisionTreeSceneAdapter
from interactive_ml_labs.gradient_scene import GradientDescentSceneAdapter
from interactive_ml_labs.knn_scene import KNNVoteMapSceneAdapter
from interactive_ml_labs.logistic_scene import LogisticRegressionSceneAdapter
from interactive_ml_labs.pygame_app import ScreenName, UnifiedAppShell
from interactive_ml_labs.random_forest_scene import RandomForestSceneAdapter
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


class FixedSizeMouseScene:
    """Tiny fixed-size scene double for input scaling tests."""

    fixed_scene_size = (100, 50)

    def __init__(self) -> None:
        """Create an empty recorder."""
        self.positions: list[tuple[int, int]] = []

    def handle_event(self, event: object) -> SceneCommand:
        """Record mouse positions passed to the scene."""
        if isinstance(event, pygame.event.Event) and hasattr(event, "pos"):
            self.positions.append(event.pos)
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the scene."""
        _ = surface


class CountingScene:
    """Tiny scene double used by restart tests."""

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        _ = event
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the scene."""
        _ = surface


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


def test_shell_starts_gradient_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real Gradient scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, GradientDescentSceneAdapter)
        assert app.screen_name == ScreenName.DEMO
    finally:
        pygame.quit()


def test_shell_starts_knn_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real k-NN scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["knn_vote_map"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, KNNVoteMapSceneAdapter)
        assert app.screen_name == ScreenName.DEMO
    finally:
        pygame.quit()


def test_shell_starts_logistic_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real Logistic Regression scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["logistic_regression_boundary_lab"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, LogisticRegressionSceneAdapter)
        assert app.screen_name == ScreenName.DEMO
    finally:
        pygame.quit()


def test_shell_starts_decision_tree_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real Decision Tree scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["decision_tree_splitter"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, DecisionTreeSceneAdapter)
        assert app.screen_name == ScreenName.DEMO
    finally:
        pygame.quit()


def test_shell_starts_random_forest_scene_from_manifest(monkeypatch) -> None:
    """The shell should start the real Random Forest scene from its manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["random_forest_bagging_lab"]
        app._start_demo()

        assert isinstance(app.scene_manager.current, RandomForestSceneAdapter)
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


def test_shell_help_overlay_uses_columns_for_demo_controls(monkeypatch) -> None:
    """Help overlay should split goals and controls into columns on wide screens."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    wrapped_items: list[tuple[str, tuple[int, int]]] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = text, position, font, color

    def capture_wrapped(
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        _ = width, color
        wrapped_items.append((text, position))
        return position[1] + font.get_linesize()

    try:
        app.selected_demo = DEMO_BY_ID["random_forest_bagging_lab"]
        app.screen_name = ScreenName.DEMO
        app.context.settings.language = "pl"
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_help_overlay()

        goal_positions = [
            position
            for text, position in wrapped_items
            if text.startswith("- Porównuj single tree baseline")
        ]
        control_positions = [
            position
            for text, position in wrapped_items
            if text.startswith(("- Up / Down:", "- W / S:", "- B / V:"))
        ]

        assert goal_positions
        assert control_positions
        assert all(position[0] < 500 for position in goal_positions)
        assert all(position[0] > 600 for position in control_positions)
        assert max(position[1] for position in control_positions) < 620
    finally:
        pygame.quit()


def test_shell_intro_uses_columns_for_long_demo_controls(monkeypatch) -> None:
    """Intro controls should not overlap objectives for demos with many shortcuts."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    wrapped_items: list[tuple[str, tuple[int, int]]] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = text, position, font, color

    def capture_wrapped(
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        _ = width, color
        wrapped_items.append((text, position))
        return position[1] + font.get_linesize()

    try:
        app.selected_demo = DEMO_BY_ID["decision_tree_splitter"]
        app.screen_name = ScreenName.INTRO
        app.context.settings.language = "pl"
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_intro()

        control_positions = [
            position
            for text, position in wrapped_items
            if text.startswith(("- M:", "- Up / Down:", "- Left / Right:", "- D:", "- G:"))
        ]

        assert control_positions
        assert all(position[0] > 600 for position in control_positions)
        assert max(position[1] for position in control_positions) < 670
    finally:
        pygame.quit()


def test_shell_demo_selection_renders_selected_demo_details(monkeypatch) -> None:
    """Demo selection should show manifest details for the highlighted demo."""
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
        _ = position, width, font, color
        wrapped_text.append(text)
        return position[1] + 24

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        app.selected_index = 2
        app._draw_wrapped = capture_wrapped

        app._render_demos()

        detail_text = " ".join(wrapped_text)
        assert "Logistic Regression Boundary Lab" in detail_text
        assert "Probabilities, thresholds" in detail_text
        assert "classification, probability" in detail_text
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


def test_shell_pause_restart_recreates_current_demo(monkeypatch) -> None:
    """Pause menu Restart should replace the active scene and resume the demo."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))
    created_scenes: list[CountingScene] = []

    def create_scene(context: object) -> CountingScene:
        _ = context
        scene = CountingScene()
        created_scenes.append(scene)
        return scene

    try:
        app.selected_demo = DEMO_BY_ID["boosting_mistake_lab"]
        app.selected_demo = app.selected_demo.__class__(
            id=app.selected_demo.id,
            level=app.selected_demo.level,
            title=app.selected_demo.title,
            summary=app.selected_demo.summary,
            objectives=app.selected_demo.objectives,
            controls=app.selected_demo.controls,
            create_scene=create_scene,
            difficulty=app.selected_demo.difficulty,
            tags=app.selected_demo.tags,
        )
        app._start_demo()
        first_scene = app.scene_manager.current
        app._open_pause()
        app.selected_index = 2
        app._activate_selected()

        assert len(created_scenes) == 2
        assert app.scene_manager.current is not first_scene
        assert app.screen_name == ScreenName.DEMO
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
        assert app.context.settings.fullscreen_enabled is True

        app.selected_index = 2
        app._activate_selected()
        assert app.context.settings.adaptive_window_enabled is True

        app.selected_index = 3
        app._activate_selected()
        assert app.context.settings.fixed_scene_scaling_enabled is False
    finally:
        pygame.quit()


def test_shell_settings_fullscreen_recreates_display_mode(monkeypatch) -> None:
    """Fullscreen toggle should immediately recreate the display surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))
    calls: list[tuple[tuple[int, int], int]] = []

    def record_set_mode(size: tuple[int, int], flags: int = 0) -> pygame.Surface:
        calls.append((size, flags))
        return pygame.Surface(size)

    try:
        monkeypatch.setattr(pygame.display, "set_mode", record_set_mode)
        app.screen_name = ScreenName.SETTINGS
        app.selected_index = 1
        app._activate_selected()

        assert app.context.settings.fullscreen_enabled is True
        assert calls[-1] == ((640, 360), pygame.FULLSCREEN)

        app._activate_selected()

        assert app.context.settings.fullscreen_enabled is False
        assert calls[-1] == ((640, 360), 0)
    finally:
        pygame.quit()


def test_shell_settings_back_returns_to_previous_screen(monkeypatch) -> None:
    """Settings Back should return to the screen that opened settings."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.DEMOS
        app._open_settings()
        app.selected_index = 5
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


def test_shell_maps_mouse_events_to_fixed_scene_coordinates(monkeypatch) -> None:
    """The shell should pass logical mouse positions to scaled fixed-size scenes."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(
        settings=AppSettings(
            resolution=(200, 200),
            fixed_scene_scaling_enabled=True,
        ),
    )
    scene = FixedSizeMouseScene()
    app.scene_manager.replace(scene)

    try:
        centered_click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": (100, 100)},
        )
        letterbox_click = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"button": 1, "pos": (100, 40)},
        )

        app._handle_active_demo_event(centered_click)
        app._handle_active_demo_event(letterbox_click)

        assert scene.positions == [(50, 25)]
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
