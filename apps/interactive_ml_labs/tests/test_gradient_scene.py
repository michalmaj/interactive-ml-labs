"""Tests for the unified Gradient Descent scene adapter."""

import pygame
from gradient_descent_playground.renderer import WINDOW_SIZE
from interactive_ml_labs.gradient_scene import (
    GRADIENT_LESSON_ID,
    LEARNING_RATE_TASK_ID,
    LOSS_DROP_TASK_ID,
    GradientDescentSceneAdapter,
    create_gradient_descent_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_gradient_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The Gradient adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gradient_descent_scene(AppContext())

        assert isinstance(scene, GradientDescentSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == WINDOW_SIZE
    finally:
        pygame.quit()


def test_gradient_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone Gradient escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gradient_descent_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_gradient_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped Gradient demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_gradient_descent_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_gradient_scene_adapter_updates_and_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should update and draw without presenting its own frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gradient_descent_scene(AppContext())
        surface = pygame.Surface(WINDOW_SIZE)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        scene.update(0.20)
        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_gradient_scene_adapter_completes_learning_rate_task(monkeypatch) -> None:
    """Changing learning rate should complete the first guided lesson task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = GRADIENT_LESSON_ID
        scene = create_gradient_descent_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        progress = context.progress.lessons[GRADIENT_LESSON_ID]
        assert LEARNING_RATE_TASK_ID in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_gradient_scene_adapter_completes_loss_drop_task_and_lesson(monkeypatch) -> None:
    """Reducing loss after several steps should complete the loss task and lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = GRADIENT_LESSON_ID
        scene = create_gradient_descent_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        for _ in range(3):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        progress = context.progress.lessons[GRADIENT_LESSON_ID]
        assert progress.completed_task_ids >= {LEARNING_RATE_TASK_ID, LOSS_DROP_TASK_ID}
        assert progress.completed is True
    finally:
        pygame.quit()


def test_gradient_scene_adapter_ignores_tasks_outside_guided_lesson(monkeypatch) -> None:
    """Standalone demo use should not mutate guided lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_gradient_descent_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        assert GRADIENT_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
