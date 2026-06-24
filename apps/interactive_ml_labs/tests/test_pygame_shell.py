"""Smoke tests for the unified Pygame shell."""

from __future__ import annotations

from types import SimpleNamespace

import pygame
from interactive_ml_labs import DEMO_BY_ID, LEARNING_PATH_MANIFESTS, LESSON_BY_ID, demos_for_level
from interactive_ml_labs.boosting_scene import (
    ADVANCE_ROUNDS_TASK_ID,
    BOOSTING_LESSON_ID,
    BoostingMistakeLabSceneAdapter,
)
from interactive_ml_labs.clustering_scene import (
    CLUSTERING_LESSON_ID,
    COMPARE_ALGORITHMS_TASK_ID,
)
from interactive_ml_labs.decision_tree_scene import DecisionTreeSceneAdapter
from interactive_ml_labs.distance_metrics_scene import (
    DISTANCE_METRICS_LESSON_ID,
    MOVE_QUERY_TASK_ID,
)
from interactive_ml_labs.gaussian_mixture_scene import (
    COMPARE_SOFT_ASSIGNMENTS_TASK_ID,
    GAUSSIAN_MIXTURE_LESSON_ID,
)
from interactive_ml_labs.gradient_scene import GradientDescentSceneAdapter
from interactive_ml_labs.kmeans_intro_scene import (
    KMEANS_LESSON_ID,
    STEP_ITERATIONS_TASK_ID,
)
from interactive_ml_labs.knn_scene import (
    CLASSIFY_QUERY_TASK_ID,
    KNN_LESSON_ID,
    KNNVoteMapSceneAdapter,
)
from interactive_ml_labs.logistic_scene import (
    LOGISTIC_LESSON_ID,
    MOVE_BOUNDARY_TASK_ID,
    LogisticRegressionSceneAdapter,
)
from interactive_ml_labs.progress import load_app_progress
from interactive_ml_labs.pygame_app import (
    DEMO_MENU_TOP,
    DEMO_SCROLLBAR_X,
    MENU_ITEM_PITCH,
    ScreenName,
    UnifiedAppShell,
)
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


def test_shell_language_selection_opens_home_screen(monkeypatch) -> None:
    """Choosing a language should enter the app home screen."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app._select_language()

        assert app.screen_name == ScreenName.HOME
    finally:
        pygame.quit()


def test_shell_home_opens_learning_paths(monkeypatch) -> None:
    """Home screen should expose guided learning paths as the first option."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.HOME
        app.selected_index = 0

        app._activate_selected()

        assert app.screen_name == ScreenName.PATHS
    finally:
        pygame.quit()


def test_shell_learning_path_selection_opens_lessons(monkeypatch) -> None:
    """Selecting a learning path should show its ordered lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.PATHS
        app.selected_index = 0

        app._activate_selected()

        assert app.screen_name == ScreenName.LESSONS
        assert app.selected_learning_path == LEARNING_PATH_MANIFESTS[0]
        assert (
            app._current_learning_path_lessons()[0]
            == LESSON_BY_ID["error_linear_regression_line_fit"]
        )
        assert app.selected_index == 0
    finally:
        pygame.quit()


def test_shell_learning_path_selection_focuses_next_incomplete_lesson(monkeypatch) -> None:
    """Selecting a path should highlight the first incomplete lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        path = LEARNING_PATH_MANIFESTS[0]
        app.context.progress.mark_completed(path.lesson_ids[0])
        app.context.progress.mark_completed(path.lesson_ids[1])
        app.screen_name = ScreenName.PATHS
        app.selected_index = 0

        app._activate_selected()

        assert app.screen_name == ScreenName.LESSONS
        assert app.selected_index == 2
        assert app.selected_learning_path == path
    finally:
        pygame.quit()


def test_shell_completed_learning_path_selection_focuses_first_lesson(monkeypatch) -> None:
    """Completed paths should open at the first lesson for review."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        path = LEARNING_PATH_MANIFESTS[0]
        for lesson_id in path.lesson_ids:
            app.context.progress.mark_completed(lesson_id)
        app.screen_name = ScreenName.PATHS
        app.selected_index = 0

        app._activate_selected()

        assert app.screen_name == ScreenName.LESSONS
        assert app.selected_index == 0
    finally:
        pygame.quit()


def test_shell_lesson_selection_opens_demo_intro(monkeypatch) -> None:
    """Selecting a lesson should reuse the existing demo intro flow."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 1

        app._activate_selected()

        assert app.screen_name == ScreenName.INTRO
        assert app.selected_lesson == LESSON_BY_ID["error_gradient_descent"]
        assert app.selected_demo == DEMO_BY_ID["gradient_descent_playground"]
        assert app.context.selected_demo_id == "gradient_descent_playground"
        assert app.context.selected_lesson_id == "error_gradient_descent"
        assert app.context.current_level == 1
    finally:
        pygame.quit()


def test_shell_lesson_selection_persists_started_progress(monkeypatch, tmp_path) -> None:
    """Selecting a lesson should mark it as started in persisted progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 1

        app._activate_selected()
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert loaded_progress.lessons["error_gradient_descent"].started is True
    finally:
        pygame.quit()


def test_shell_open_theory_persists_theory_visit_for_selected_lesson(
    monkeypatch,
    tmp_path,
) -> None:
    """Opening theory from a selected lesson should persist theory progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_lesson = LESSON_BY_ID["error_gradient_descent"]
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.screen_name = ScreenName.INTRO

        app._open_theory()
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert loaded_progress.lessons["error_gradient_descent"].started is True
        assert loaded_progress.lessons["error_gradient_descent"].theory_visited is True
    finally:
        pygame.quit()


def test_shell_explicit_settings_keep_progress_in_memory(monkeypatch, tmp_path) -> None:
    """Explicit settings should not cause user progress files to be written in tests."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.progress_path = tmp_path / "progress.json"
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS

        app._activate_selected()

        assert app.context.progress.lessons["error_linear_regression_line_fit"].started is True
        assert not app.progress_path.exists()
    finally:
        pygame.quit()


def test_shell_persists_task_progress_from_active_demo(monkeypatch, tmp_path) -> None:
    """Task progress reported by an active scene should be saved by the shell."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 1
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            "find_stable_learning_rate"
            in loaded_progress.lessons["error_gradient_descent"].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_linear_regression_task_progress(monkeypatch, tmp_path) -> None:
    """Linear Regression task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 0
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            "balance_residuals"
            in loaded_progress.lessons["error_linear_regression_line_fit"].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_logistic_regression_task_progress(monkeypatch, tmp_path) -> None:
    """Logistic Regression task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 2
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            MOVE_BOUNDARY_TASK_ID in loaded_progress.lessons[LOGISTIC_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_boosting_task_progress(monkeypatch, tmp_path) -> None:
    """Boosting task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 3
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            ADVANCE_ROUNDS_TASK_ID in loaded_progress.lessons[BOOSTING_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_distance_metrics_task_progress(monkeypatch, tmp_path) -> None:
    """Distance Metrics task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[1]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 0
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            MOVE_QUERY_TASK_ID
            in loaded_progress.lessons[DISTANCE_METRICS_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_knn_task_progress(monkeypatch, tmp_path) -> None:
    """k-NN task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[1]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 1
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert CLASSIFY_QUERY_TASK_ID in loaded_progress.lessons[KNN_LESSON_ID].completed_task_ids
    finally:
        pygame.quit()


def test_shell_persists_kmeans_task_progress(monkeypatch, tmp_path) -> None:
    """K-Means task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[1]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 2
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            STEP_ITERATIONS_TASK_ID in loaded_progress.lessons[KMEANS_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_clustering_task_progress(monkeypatch, tmp_path) -> None:
    """Clustering task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[1]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 3
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            COMPARE_ALGORITHMS_TASK_ID
            in loaded_progress.lessons[CLUSTERING_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_persists_gaussian_mixture_task_progress(monkeypatch, tmp_path) -> None:
    """Gaussian Mixture task progress should be saved from active scene events."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    settings_path = tmp_path / "settings.json"
    app = UnifiedAppShell(settings_path=settings_path)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[1]
        app.screen_name = ScreenName.LESSONS
        app.selected_index = 4
        app._activate_selected()
        app._start_demo()

        app._handle_active_demo_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        loaded_progress = load_app_progress(tmp_path / "progress.json")

        assert (
            COMPARE_SOFT_ASSIGNMENTS_TASK_ID
            in loaded_progress.lessons[GAUSSIAN_MIXTURE_LESSON_ID].completed_task_ids
        )
    finally:
        pygame.quit()


def test_shell_lesson_task_labels_reflect_progress(monkeypatch) -> None:
    """Lesson task labels should show completed and pending task state."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        lesson = LESSON_BY_ID["error_gradient_descent"]
        app.context.progress.complete_task(lesson.id, "find_stable_learning_rate")

        assert app._lesson_task_summary(lesson) == "Tasks: 1/2 completed"
        assert (
            app._lesson_task_label(lesson, "find_stable_learning_rate", "Find learning rate")
            == "[x] Find learning rate"
        )
        assert app._lesson_task_label(lesson, "observe_loss_drop", "Observe loss") == (
            "[ ] Observe loss"
        )
    finally:
        pygame.quit()


def test_shell_lesson_menu_labels_reflect_progress(monkeypatch) -> None:
    """Lesson menu labels should show compact lesson and task progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        lesson = LESSON_BY_ID["error_gradient_descent"]

        assert app._lesson_menu_label(lesson) == "[ ] Let an algorithm reduce loss"

        app.context.progress.mark_started(lesson.id)
        assert app._lesson_menu_label(lesson) == "[..] Let an algorithm reduce loss"

        app.context.progress.complete_task(lesson.id, "find_stable_learning_rate")
        assert app._lesson_menu_label(lesson) == "[1/2] Let an algorithm reduce loss"

        app.context.progress.mark_completed(lesson.id)
        assert app._lesson_menu_label(lesson) == "[x] Let an algorithm reduce loss"
    finally:
        pygame.quit()


def test_shell_lessons_screen_renders_progress_markers(monkeypatch) -> None:
    """Lesson selection should pass progress-aware labels to the menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))
    menu_labels: list[str] = []

    def capture_menu(
        labels: list[str],
        *,
        top: int,
        width: int = 640,
    ) -> None:
        _ = top, width
        menu_labels.extend(labels)

    try:
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.context.progress.complete_task("error_gradient_descent", "find_stable_learning_rate")
        app._draw_menu = capture_menu

        app._render_lessons()

        assert "[1/2] Let an algorithm reduce loss" in menu_labels
    finally:
        pygame.quit()


def test_shell_lesson_badge_label_reflects_completion(monkeypatch) -> None:
    """Lesson badge label should distinguish locked and unlocked badges."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        lesson = LESSON_BY_ID["error_gradient_descent"]

        assert app._lesson_badge_label(lesson) == "Badge locked: Loss Navigator"

        app.context.progress.mark_completed(lesson.id)

        assert app._lesson_badge_label(lesson) == "Badge unlocked: Loss Navigator"
    finally:
        pygame.quit()


def test_shell_learning_path_progress_labels_reflect_completion(monkeypatch) -> None:
    """Learning path labels should summarize lessons and tasks."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        path = LEARNING_PATH_MANIFESTS[0]

        assert app._learning_path_progress_label(path) == "Lessons: 0/4 completed"
        assert app._learning_path_task_progress_label(path) == "Tasks: 0/8 completed"
        assert app._learning_path_status_label(path) == "Not started"
        assert app._learning_path_next_action_label(path) == (
            "Next action: start See error in a fitted line"
        )

        app.context.progress.mark_started(path.lesson_ids[0])
        app.context.progress.complete_task(path.lesson_ids[0], "move_line")
        app.context.progress.complete_task(path.lesson_ids[0], "balance_residuals")
        app.context.progress.mark_completed(path.lesson_ids[1])

        assert app._learning_path_progress_label(path) == "Lessons: 1/4 completed"
        assert app._learning_path_task_progress_label(path) == "Tasks: 2/8 completed"
        assert app._learning_path_status_label(path) == "In progress"
        assert app._learning_path_next_action_label(path) == (
            "Next action: continue See error in a fitted line"
        )

        for lesson_id in path.lesson_ids:
            app.context.progress.mark_completed(lesson_id)

        assert app._learning_path_progress_label(path) == "Lessons: 4/4 completed"
        assert app._learning_path_status_label(path) == "Path completed"
        assert app._learning_path_next_action_label(path) == (
            "Next action: review completed lessons"
        )
    finally:
        pygame.quit()


def test_shell_learning_path_details_render_progress_summary(monkeypatch) -> None:
    """Learning path details should render aggregate progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []
    wrapped_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

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
        path = LEARNING_PATH_MANIFESTS[0]
        app.context.progress.mark_completed(path.lesson_ids[0])
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_learning_path_details(path)

        assert "4 lessons" in drawn_text
        assert "Lessons: 1/4 completed" in wrapped_text
        assert "Tasks: 0/8 completed" in wrapped_text
        assert "In progress" in wrapped_text
        assert "Next action: start Let an algorithm reduce loss" in wrapped_text
    finally:
        pygame.quit()


def test_shell_learning_path_task_progress_label_localizes_polish(monkeypatch) -> None:
    """Learning path task summaries should use Polish UI copy."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.context.settings.language = "pl"
        path = LEARNING_PATH_MANIFESTS[1]
        first_lesson_id = path.lesson_ids[0]
        app.context.progress.complete_task(first_lesson_id, "move_query")

        assert app._learning_path_task_progress_label(path) == "Zadania: 1/10 ukończone"
        assert app._learning_path_next_action_label(path) == (
            "Następny krok: kontynuuj Ustal, co znaczy blisko"
        )
    finally:
        pygame.quit()


def test_shell_lesson_guidance_labels_reflect_path_order_and_prerequisites(
    monkeypatch,
) -> None:
    """Lesson guidance should show prerequisites and the next lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        path = LEARNING_PATH_MANIFESTS[0]
        app.selected_learning_path = path
        lesson = LESSON_BY_ID["error_logistic_boundary"]

        assert app._lesson_prerequisite_label(lesson) == (
            "Prerequisites: [ ] Let an algorithm reduce loss"
        )
        assert app._lesson_next_label(lesson) == "Next: Learn from previous mistakes"

        app.context.progress.mark_completed("error_gradient_descent")

        assert app._lesson_prerequisite_label(lesson) == (
            "Prerequisites: [x] Let an algorithm reduce loss"
        )
    finally:
        pygame.quit()


def test_shell_lesson_guidance_labels_handle_first_and_final_lessons(monkeypatch) -> None:
    """Guidance should handle lessons without prerequisites and final lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        path = LEARNING_PATH_MANIFESTS[0]
        app.selected_learning_path = path

        first_lesson = LESSON_BY_ID[path.lesson_ids[0]]
        final_lesson = LESSON_BY_ID[path.lesson_ids[-1]]

        assert app._lesson_prerequisite_label(first_lesson) == "Prerequisites: none"
        assert app._lesson_next_label(final_lesson) == "Next: final lesson in this path"
    finally:
        pygame.quit()


def test_shell_lesson_details_render_task_checklist(monkeypatch) -> None:
    """Lesson details panel should render task checklist state."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []
    wrapped_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

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
        lesson = LESSON_BY_ID["error_gradient_descent"]
        app.selected_learning_path = LEARNING_PATH_MANIFESTS[0]
        app.context.progress.complete_task(lesson.id, "find_stable_learning_rate")
        app.context.progress.mark_completed(lesson.id)
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_lesson_details(lesson)

        assert "Tasks: 1/2 completed" in drawn_text
        assert any(text.startswith("[x]") for text in wrapped_text)
        assert any(text.startswith("[ ]") for text in wrapped_text)
        assert "Prerequisites: [ ] See error in a fitted line" in wrapped_text
        assert "Next: Turn scores into decisions" in wrapped_text
        assert "Badge unlocked: Loss Navigator" in wrapped_text
    finally:
        pygame.quit()


def test_shell_intro_renders_selected_lesson_task_checklist(monkeypatch) -> None:
    """Intro screen should show lesson tasks when launched from a learning path."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []
    wrapped_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

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
        lesson = LESSON_BY_ID["error_gradient_descent"]
        app.selected_lesson = lesson
        app.selected_demo = DEMO_BY_ID[lesson.demo_id]
        app.context.progress.complete_task(lesson.id, "find_stable_learning_rate")
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_intro()

        assert "Lesson tasks" in drawn_text
        assert "[x] Find a stable learning rate" in wrapped_text
        assert "[ ] Observe the loss drop" in wrapped_text
        assert "Theory: not visited" in wrapped_text
        assert "Badge locked: Loss Navigator" in wrapped_text
    finally:
        pygame.quit()


def test_shell_intro_omits_lesson_tasks_for_standalone_demo(monkeypatch) -> None:
    """Standalone demo intros should keep the original demo-only layout."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

    try:
        app.selected_demo = DEMO_BY_ID["gradient_descent_playground"]
        app.selected_lesson = None
        app._draw_text = capture_text

        app._render_intro()

        assert "Lesson tasks" not in drawn_text
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


def test_level_two_and_three_intro_copy_stays_above_footer(monkeypatch) -> None:
    """Level 2 and 3 intro copy should stay clear of the shared footer."""
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

        for level in (2, 3):
            for demo in demos_for_level(level):
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


def test_level_two_and_three_help_overlay_copy_stays_inside_overlay(monkeypatch) -> None:
    """Level 2 and 3 help overlay text should stay inside the dialog body."""
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

        for level in (2, 3):
            for demo in demos_for_level(level):
                wrapped_bottoms.clear()
                app.selected_demo = demo
                app._render_help_overlay()
                assert wrapped_bottoms
                assert max(wrapped_bottoms) <= overlay_bottom
    finally:
        pygame.quit()


def test_level_one_theory_screens_render_in_polish_and_english(monkeypatch) -> None:
    """Level 1 theory screens should render without relying on external markdown files."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.screen_name = ScreenName.THEORY
        for language in ("pl", "en"):
            app.context.settings.language = language
            for demo in demos_for_level(1):
                app.selected_demo = demo
                app.theory_scroll_offset = 0
                app.theory_max_scroll = 0

                app._render_theory()

                assert app.theory_max_scroll >= 0
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


def test_shell_demo_list_hover_uses_scrolled_position_before_rerender(monkeypatch) -> None:
    """Hover selection should account for demo scroll offset immediately after wheel input."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.context.current_level = 1
        app.screen_name = ScreenName.DEMOS
        app._render_demos()

        app._handle_mouse_wheel(-10)
        scrolled_offset = app.demo_scroll_offset
        hover_position = (120, DEMO_MENU_TOP + 14)

        app._handle_mouse_motion(hover_position)

        expected_index = scrolled_offset // MENU_ITEM_PITCH
        assert scrolled_offset > 0
        assert app.selected_index == expected_index
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


def test_shell_pause_renders_selected_lesson_task_checklist(monkeypatch) -> None:
    """Pause menu should show lesson task progress during guided lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []
    wrapped_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

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
        lesson = LESSON_BY_ID["error_gradient_descent"]
        app.screen_name = ScreenName.PAUSE
        app.selected_lesson = lesson
        app.context.progress.complete_task(lesson.id, "find_stable_learning_rate")
        app._draw_text = capture_text
        app._draw_wrapped = capture_wrapped

        app._render_pause()

        assert "Lesson tasks" in drawn_text
        assert "Tasks: 1/2 completed" in wrapped_text
        assert "Theory: not visited" in wrapped_text
        assert "[x] Find a stable learning rate" in wrapped_text
        assert "[ ] Observe the loss drop" in wrapped_text
        assert "Badge locked: Loss Navigator" in wrapped_text
    finally:
        pygame.quit()


def test_shell_pause_shows_unlocked_lesson_badge(monkeypatch) -> None:
    """Pause menu should show when a guided lesson badge is unlocked."""
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
        lesson = LESSON_BY_ID["error_gradient_descent"]
        app.screen_name = ScreenName.PAUSE
        app.selected_lesson = lesson
        app.context.progress.mark_completed(lesson.id)
        app._draw_wrapped = capture_wrapped

        app._render_pause()

        assert "Badge unlocked: Loss Navigator" in wrapped_text
    finally:
        pygame.quit()


def test_shell_lesson_theory_status_label_localizes_polish(monkeypatch) -> None:
    """Theory progress labels should reflect visited state in Polish."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))

    try:
        app.context.settings.language = "pl"
        lesson = LESSON_BY_ID["error_gradient_descent"]

        assert app._lesson_theory_status_label(lesson) == "Teoria: nieprzeczytana"

        app.context.progress.mark_theory_visited(lesson.id)

        assert app._lesson_theory_status_label(lesson) == "Teoria: przeczytana"
    finally:
        pygame.quit()


def test_shell_pause_omits_lesson_tasks_for_standalone_demo(monkeypatch) -> None:
    """Standalone demo pause menu should keep the original menu-only layout."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(1280, 720)))
    drawn_text: list[str] = []

    def capture_text(
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        _ = position, font, color
        drawn_text.append(text)

    try:
        app.screen_name = ScreenName.PAUSE
        app.selected_lesson = None
        app._draw_text = capture_text

        app._render_pause()

        assert "Lesson tasks" not in drawn_text
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

        assert app.screen_name == ScreenName.HOME
    finally:
        pygame.quit()


def test_shell_home_escape_returns_to_language(monkeypatch) -> None:
    """Esc from the home screen should return to language selection."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    app = UnifiedAppShell(settings=AppSettings(resolution=(640, 360)))

    try:
        app.screen_name = ScreenName.HOME
        app._escape()

        assert app.screen_name == ScreenName.LANGUAGE
    finally:
        pygame.quit()
