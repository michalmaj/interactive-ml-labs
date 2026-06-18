"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from interactive_ml_labs import DEMO_BY_ID, demos_for_level
from interactive_ml_labs.boosting_scene import BoostingMistakeLabSceneAdapter
from interactive_ml_labs.decision_tree_scene import DecisionTreeSceneAdapter
from interactive_ml_labs.gradient_scene import GradientDescentSceneAdapter
from interactive_ml_labs.knn_scene import KNNVoteMapSceneAdapter
from interactive_ml_labs.logistic_scene import LogisticRegressionSceneAdapter
from interactive_ml_labs.pygame_app import DEMO_SCROLLBAR_X, ScreenName, UnifiedAppShell
from interactive_ml_labs.random_forest_scene import RandomForestSceneAdapter
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppSettings, save_app_settings


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


def _wrapped_text_bottom(
    text: str,
    position: tuple[int, int],
    width: int,
    font: pygame.font.Font,
) -> int:
    """Return the bottom y coordinate for text wrapped like the shell does."""
    words = text.split()
    line = ""
    _, y = position

    for word in words:
        candidate = f"{line} {word}".strip()
        if font.size(candidate)[0] <= width:
            line = candidate
            continue

        y += font.get_linesize()
        line = word

    if line:
        y += font.get_linesize()

    return y


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


def test_level_one_intro_copy_stays_above_footer(monkeypatch) -> None:
    """Level 1 intro objectives and controls should not collide with the footer."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    bottom_limit = app._content_bottom()
    wrapped_bottoms: list[int] = []

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
        _ = color
        y = _wrapped_text_bottom(text, position, width, font)
        wrapped_bottoms.append(y)
        return y

    try:
        app.context.settings.language = "pl"
        app.screen_name = ScreenName.INTRO
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        for demo in demos_for_level(1):
            wrapped_bottoms.clear()
            app.selected_demo = demo
            app._render_intro()
            assert wrapped_bottoms
            assert max(wrapped_bottoms) <= bottom_limit
    finally:
        pygame.quit()


def test_level_one_help_overlay_copy_stays_inside_overlay(monkeypatch) -> None:
    """Level 1 help overlay text should stay inside the dialog body."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    overlay_bottom = 720 - min(90, max(32, 720 // 10)) - 32
    wrapped_bottoms: list[int] = []

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
        _ = color
        y = _wrapped_text_bottom(text, position, width, font)
        wrapped_bottoms.append(y)
        return y

    try:
        app.context.settings.language = "pl"
        app.screen_name = ScreenName.INTRO
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        for demo in demos_for_level(1):
            wrapped_bottoms.clear()
            app.selected_demo = demo
            app._render_help_overlay()
            assert wrapped_bottoms
            assert max(wrapped_bottoms) <= overlay_bottom
    finally:
        pygame.quit()


def test_shell_opens_theory_screen_from_intro(monkeypatch) -> None:
    """The intro screen should expose the generated in-app theory screen."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.screen_name = ScreenName.INTRO

        app._handle_keydown(pygame.K_t)

        assert app.screen_name == ScreenName.THEORY
        assert app.theory_return_screen == ScreenName.INTRO
    finally:
        pygame.quit()


def test_shell_theory_screen_renders_manifest_sections(monkeypatch) -> None:
    """Theory screen should render lesson sections from the manifest."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    rendered_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        rendered_text.append(text)

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.screen_name = ScreenName.THEORY
        app.context.settings.language = "pl"
        app._draw_text = capture_text

        app._render_theory()

        theory_text = " ".join(rendered_text)
        assert "Gradient descent" in theory_text
        assert "learning rate" in theory_text
        assert "loss" in theory_text
        assert "Mini-zadania" in theory_text

        rendered_text.clear()
        app.theory_scroll_offset = app.theory_max_scroll
        app._render_theory()

        theory_text = " ".join(rendered_text)
        assert "Słowniczek" in theory_text
        assert "gradient:" in theory_text
    finally:
        pygame.quit()


def test_shell_theory_content_stays_above_footer(monkeypatch) -> None:
    """Long theory content should not draw into the shared footer area."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    rendered_text: list[tuple[str, tuple[int, int]]] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = font, color
        rendered_text.append((text, position))

    try:
        app.selected_demo = DEMO_BY_ID["random_forest_bagging_lab"]
        app.screen_name = ScreenName.THEORY
        app.context.settings.language = "pl"
        app._draw_text = capture_text

        app._render_theory()

        footer_y = app._footer_y()
        content_bottom = app._content_bottom()
        content_positions = [
            position
            for text, position in rendered_text
            if position[1] < footer_y and not text.startswith(("Enter:", "Esc/Backspace:"))
        ]

        assert content_positions
        assert max(position[1] for position in content_positions) <= content_bottom
        assert max(position[1] for position in content_positions) < footer_y
    finally:
        pygame.quit()


def test_shell_theory_mouse_wheel_scrolls_content(monkeypatch) -> None:
    """Mouse wheel should scroll the generated theory screen."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(900, 520)))

    try:
        app.selected_demo = DEMO_BY_ID["random_forest_bagging_lab"]
        app.screen_name = ScreenName.THEORY

        app._render_theory()
        assert app.theory_max_scroll > 0

        app._handle_mouse_wheel(-1)
        assert app.theory_scroll_offset > 0

        app._handle_mouse_wheel(100)
        assert app.theory_scroll_offset == 0
    finally:
        pygame.quit()


def test_shell_open_theory_resets_scroll_position(monkeypatch) -> None:
    """Opening theory should start at the top of the lesson notes."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(900, 520)))

    try:
        app.selected_demo = DEMO_BY_ID["random_forest_bagging_lab"]
        app.screen_name = ScreenName.INTRO
        app.theory_scroll_offset = 120

        app._open_theory()

        assert app.screen_name == ScreenName.THEORY
        assert app.theory_scroll_offset == 0
    finally:
        pygame.quit()


def test_shell_theory_enter_starts_demo_from_intro(monkeypatch) -> None:
    """Enter on theory should start the demo when the student came from intro."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.screen_name = ScreenName.THEORY
        app.theory_return_screen = ScreenName.INTRO

        app._activate_selected()

        assert app.screen_name == ScreenName.DEMO
        assert isinstance(app.scene_manager.current, GradientDescentSceneAdapter)
    finally:
        pygame.quit()


def test_shell_theory_escape_returns_to_pause(monkeypatch) -> None:
    """Esc on theory should return to pause when opened from the pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.screen_name = ScreenName.PAUSE
        app.selected_index = 2

        app._activate_selected()
        assert app.screen_name == ScreenName.THEORY

        app._escape()
        assert app.screen_name == ScreenName.PAUSE
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
        app.selected_index = tuple(demo.id for demo in demos_for_level(1)).index(
            "logistic_regression_boundary_lab"
        )
        app._draw_wrapped = capture_wrapped

        app._render_demos()

        detail_text = " ".join(wrapped_text)
        assert "Logistic Regression Boundary Lab" in detail_text
        assert "Probabilities, thresholds" in detail_text
        assert "classification, probability" in detail_text
    finally:
        pygame.quit()


def test_shell_demo_list_scrolls_selected_item_above_footer(monkeypatch) -> None:
    """Long demo lists should keep the selected item above the shared footer."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        for _ in range(len(demos_for_level(1)) - 1):
            app._move_down()

        app._render_demos()

        assert app.demo_scroll_offset > 0
        assert app.menu_items
        assert max(item.rect.bottom for item in app.menu_items) <= app._content_bottom()
        assert max(item.rect.bottom for item in app.menu_items) < app._footer_y()
    finally:
        pygame.quit()


def test_shell_demo_list_mouse_wheel_scrolls_and_draws_indicator(monkeypatch) -> None:
    """Mouse wheel should scroll long demo lists and expose a scrollbar."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    scrollbars: list[tuple[int, int, int, int]] = []
    original_draw_rect = pygame.draw.rect

    def capture_rect(
        surface: pygame.Surface,
        color: tuple[int, int, int],
        rect: pygame.Rect,
        width: int = 0,
        border_radius: int = 0,
        *args: object,
        **kwargs: object,
    ) -> pygame.Rect:
        if rect.x == 598 and rect.width == 4:
            scrollbars.append((rect.x, rect.y, rect.width, rect.height))
        return original_draw_rect(surface, color, rect, width, border_radius, *args, **kwargs)

    monkeypatch.setattr(pygame.draw, "rect", capture_rect)

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        app._render_demos()

        assert app.demo_max_scroll > 0

        app._handle_mouse_wheel(-1)
        app._render_demos()

        assert app.demo_scroll_offset > 0
        assert scrollbars
    finally:
        pygame.quit()


def test_shell_demo_scrollbar_track_click_jumps_list(monkeypatch) -> None:
    """Clicking the demo scrollbar track should jump the list near that position."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        app._render_demos()
        bottom = app._content_bottom()
        target_y = bottom - 8

        app._handle_mouse_click((DEMO_SCROLLBAR_X, target_y))

        assert app.demo_scroll_offset > 0
        assert app.demo_scroll_offset == app.demo_max_scroll
        assert app.selected_index > 0
    finally:
        pygame.quit()


def test_shell_demo_scrollbar_thumb_drag_scrolls_list(monkeypatch) -> None:
    """Dragging the demo scrollbar thumb should update the scroll offset until release."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        app._render_demos()
        top = 190
        bottom = app._content_bottom()
        _, thumb_rect = app._demo_scrollbar_hit_rects(top, bottom)

        app._handle_mouse_click((DEMO_SCROLLBAR_X, thumb_rect.centery))
        app._handle_mouse_motion((DEMO_SCROLLBAR_X, bottom - 8))

        assert app.demo_scrollbar_dragging is True
        assert app.demo_scroll_offset == app.demo_max_scroll
        assert app.selected_index > 0

        app._handle_mouse_release()

        assert app.demo_scrollbar_dragging is False
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


def test_shell_pause_menu_stays_above_footer(monkeypatch) -> None:
    """Pause menu options should not collide with the shared footer."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.screen_name = ScreenName.PAUSE

        app._render_pause()

        assert app.menu_items
        assert max(item.rect.bottom for item in app.menu_items) < app._footer_y()
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
            theory=app.selected_demo.theory,
        )
        app._start_demo()
        first_scene = app.scene_manager.current
        app._open_pause()
        app.selected_index = 3
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


def test_shell_loads_and_saves_persisted_settings(monkeypatch, tmp_path) -> None:
    """The shell should load settings from disk and persist user changes."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    save_app_settings(
        AppSettings(
            language="pl",
            resolution=(640, 360),
            adaptive_window_enabled=True,
            fixed_scene_scaling_enabled=False,
        ),
        settings_path,
    )
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        assert app.context.settings.language == "pl"
        assert app.context.settings.adaptive_window_enabled is True
        assert app.context.settings.fixed_scene_scaling_enabled is False

        app._handle_keydown(pygame.K_l)
        loaded = settings_path.read_text(encoding="utf-8")

        assert app.context.settings.language == "en"
        assert '"language": "en"' in loaded

        app.screen_name = ScreenName.SETTINGS
        app.selected_index = 1
        app._activate_selected()
        loaded = settings_path.read_text(encoding="utf-8")

        assert app.context.settings.fullscreen_enabled is True
        assert '"fullscreen_enabled": true' in loaded
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
