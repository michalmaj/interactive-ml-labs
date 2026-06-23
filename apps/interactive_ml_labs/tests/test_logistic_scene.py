"""Tests for the unified Logistic Regression scene adapter."""

import pygame
from interactive_ml_labs.logistic_scene import (
    COMPARE_POINTS_TASK_ID,
    LOGISTIC_LESSON_ID,
    LogisticRegressionSceneAdapter,
    MOVE_BOUNDARY_TASK_ID,
    create_logistic_regression_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from logistic_regression_boundary_lab.renderer import WINDOW_SIZE


def test_logistic_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The Logistic Regression adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_logistic_regression_scene(AppContext())

        assert isinstance(scene, LogisticRegressionSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == WINDOW_SIZE
    finally:
        pygame.quit()


def test_logistic_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone Logistic Regression escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_logistic_regression_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_logistic_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped Logistic Regression demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_logistic_regression_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_logistic_scene_adapter_updates_and_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should update and draw without presenting its own frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_logistic_regression_scene(AppContext())
        surface = pygame.Surface(WINDOW_SIZE)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        scene.update(0.20)
        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_logistic_scene_adapter_records_boundary_task_for_guided_lesson(
    monkeypatch,
) -> None:
    """Changing a Logistic Regression parameter should complete the boundary task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = LOGISTIC_LESSON_ID
        scene = create_logistic_regression_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))

        progress = context.progress.lessons[LOGISTIC_LESSON_ID]
        assert MOVE_BOUNDARY_TASK_ID in progress.completed_task_ids
        assert COMPARE_POINTS_TASK_ID not in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_logistic_scene_adapter_records_compare_task_and_completes_lesson(
    monkeypatch,
) -> None:
    """Training interaction should complete the compare task and then the lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = LOGISTIC_LESSON_ID
        scene = create_logistic_regression_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        progress = context.progress.lessons[LOGISTIC_LESSON_ID]
        assert MOVE_BOUNDARY_TASK_ID in progress.completed_task_ids
        assert COMPARE_POINTS_TASK_ID in progress.completed_task_ids
        assert progress.completed is True
    finally:
        pygame.quit()


def test_logistic_scene_adapter_ignores_task_progress_outside_guided_lesson(
    monkeypatch,
) -> None:
    """Standalone Logistic Regression interactions should not write lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_logistic_regression_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        assert LOGISTIC_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
