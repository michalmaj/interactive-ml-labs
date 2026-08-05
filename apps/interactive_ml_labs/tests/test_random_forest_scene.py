"""Tests for the unified Random Forest scene adapter."""

import pygame
from interactive_ml_labs.random_forest_scene import (
    COMPARE_FOREST_VOTE_TASK_ID,
    INSPECT_FOREST_CONFIDENCE_TASK_ID,
    RANDOM_FOREST_LESSON_ID,
    RandomForestSceneAdapter,
    create_random_forest_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from random_forest_bagging_lab.renderer import WINDOW_SIZE


def test_random_forest_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The Random Forest adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_random_forest_scene(AppContext())

        assert isinstance(scene, RandomForestSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == WINDOW_SIZE
    finally:
        pygame.quit()


def test_random_forest_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone Random Forest escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_random_forest_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_random_forest_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped Random Forest demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_random_forest_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_random_forest_scene_adapter_reports_guided_lesson_progress(monkeypatch) -> None:
    """Guided mode should complete forest tasks from vote and confidence controls."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=RANDOM_FOREST_LESSON_ID)
        scene = create_random_forest_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        progress = context.progress.lessons[RANDOM_FOREST_LESSON_ID]
        assert progress.completed_task_ids == {COMPARE_FOREST_VOTE_TASK_ID}
        assert progress.completed is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))

        progress = context.progress.lessons[RANDOM_FOREST_LESSON_ID]
        assert progress.completed_task_ids == {
            COMPARE_FOREST_VOTE_TASK_ID,
            INSPECT_FOREST_CONFIDENCE_TASK_ID,
        }
        assert progress.completed is True
    finally:
        pygame.quit()


def test_random_forest_scene_adapter_standalone_does_not_mutate_guided_progress(
    monkeypatch,
) -> None:
    """Standalone Random Forest interactions should not complete guided lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_random_forest_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))

        assert context.progress.lessons == {}
    finally:
        pygame.quit()


def test_random_forest_scene_adapter_updates_and_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should update and draw without presenting its own frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_random_forest_scene(AppContext())
        surface = pygame.Surface(WINDOW_SIZE)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        scene.update(0.20)
        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
