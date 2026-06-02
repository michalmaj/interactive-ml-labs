"""Tests for the unified Random Forest scene adapter."""

import pygame
from interactive_ml_labs.random_forest_scene import (
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
