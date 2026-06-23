"""Tests for the unified Boosting Mistake Lab scene adapter."""

import pygame
from interactive_ml_labs.boosting_scene import (
    ADVANCE_ROUNDS_TASK_ID,
    BOOSTING_LESSON_ID,
    BoostingMistakeLabSceneAdapter,
    CONNECT_WEIGHT_TASK_ID,
    create_boosting_mistake_lab_scene,
)
from interactive_ml_labs.display import BOOSTING_FIXED_SCENE_SIZE
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_boosting_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The Boosting adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_boosting_mistake_lab_scene(AppContext())

        assert isinstance(scene, BoostingMistakeLabSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == BOOSTING_FIXED_SCENE_SIZE
    finally:
        pygame.quit()


def test_boosting_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone Boosting escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_boosting_mistake_lab_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_boosting_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped Boosting demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_boosting_mistake_lab_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_boosting_scene_adapter_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should draw without presenting its own display frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_boosting_mistake_lab_scene(AppContext())
        surface = pygame.Surface(BOOSTING_FIXED_SCENE_SIZE)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_boosting_scene_adapter_records_round_task_for_guided_lesson(
    monkeypatch,
) -> None:
    """Changing the selected boosting stage should complete the round task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = BOOSTING_LESSON_ID
        scene = create_boosting_mistake_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        progress = context.progress.lessons[BOOSTING_LESSON_ID]
        assert ADVANCE_ROUNDS_TASK_ID in progress.completed_task_ids
        assert CONNECT_WEIGHT_TASK_ID not in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_boosting_scene_adapter_records_weight_task_and_completes_lesson(
    monkeypatch,
) -> None:
    """Confidence view should complete the weight task and then the lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = BOOSTING_LESSON_ID
        scene = create_boosting_mistake_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))

        progress = context.progress.lessons[BOOSTING_LESSON_ID]
        assert ADVANCE_ROUNDS_TASK_ID in progress.completed_task_ids
        assert CONNECT_WEIGHT_TASK_ID in progress.completed_task_ids
        assert progress.completed is True
    finally:
        pygame.quit()


def test_boosting_scene_adapter_ignores_task_progress_outside_guided_lesson(
    monkeypatch,
) -> None:
    """Standalone Boosting interactions should not write lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_boosting_mistake_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))

        assert BOOSTING_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
