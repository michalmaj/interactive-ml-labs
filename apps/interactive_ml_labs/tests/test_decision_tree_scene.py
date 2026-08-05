"""Tests for the unified Decision Tree scene adapter."""

import pygame
from decision_tree_splitter.renderer import WINDOW_SIZE
from interactive_ml_labs.decision_tree_scene import (
    DECISION_TREE_LESSON_ID,
    INSPECT_FIRST_SPLIT_TASK_ID,
    MOVE_TREE_SPLIT_TASK_ID,
    DecisionTreeSceneAdapter,
    create_decision_tree_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_decision_tree_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The Decision Tree adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_decision_tree_scene(AppContext())

        assert isinstance(scene, DecisionTreeSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == WINDOW_SIZE
    finally:
        pygame.quit()


def test_decision_tree_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone Decision Tree escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_decision_tree_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_decision_tree_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped Decision Tree demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_decision_tree_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_decision_tree_scene_adapter_reports_guided_lesson_progress(monkeypatch) -> None:
    """Guided mode should complete tree tasks from meaningful split controls."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=DECISION_TREE_LESSON_ID)
        scene = create_decision_tree_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        progress = context.progress.lessons[DECISION_TREE_LESSON_ID]
        assert progress.completed_task_ids == {MOVE_TREE_SPLIT_TASK_ID}
        assert progress.completed is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        progress = context.progress.lessons[DECISION_TREE_LESSON_ID]
        assert progress.completed_task_ids == {
            MOVE_TREE_SPLIT_TASK_ID,
            INSPECT_FIRST_SPLIT_TASK_ID,
        }
        assert progress.completed is True
    finally:
        pygame.quit()


def test_decision_tree_scene_adapter_standalone_does_not_mutate_guided_progress(
    monkeypatch,
) -> None:
    """Standalone Decision Tree interactions should not complete guided lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_decision_tree_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        assert context.progress.lessons == {}
    finally:
        pygame.quit()


def test_decision_tree_scene_adapter_updates_and_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should update and draw without presenting its own frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_decision_tree_scene(AppContext())
        surface = pygame.Surface(WINDOW_SIZE)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.update(0.20)
        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
