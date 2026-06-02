"""Tests for the unified k-NN Vote Map scene adapter."""

import pygame
from interactive_ml_labs.knn_scene import KNNVoteMapSceneAdapter, create_knn_vote_map_scene
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from knn_vote_map.renderer import WINDOW_SIZE


def test_knn_scene_adapter_exposes_fixed_scene_contract(monkeypatch) -> None:
    """The k-NN adapter should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_knn_vote_map_scene(AppContext())

        assert isinstance(scene, KNNVoteMapSceneAdapter)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == WINDOW_SIZE
    finally:
        pygame.quit()


def test_knn_scene_adapter_translates_escape_to_pause(monkeypatch) -> None:
    """Standalone k-NN escape handling should open the shell pause menu."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_knn_vote_map_scene(AppContext())
        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

        assert command.kind == SceneCommandKind.PAUSE
    finally:
        pygame.quit()


def test_knn_scene_adapter_passes_shell_language(monkeypatch) -> None:
    """The wrapped k-NN demo should use the shell language setting."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_knn_vote_map_scene(context)

        assert scene._scene._renderer._language == "pl"
    finally:
        pygame.quit()


def test_knn_scene_adapter_updates_and_renders_to_shell_surface(monkeypatch) -> None:
    """The adapter should update and draw without presenting its own frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_knn_vote_map_scene(AppContext())
        surface = pygame.Surface(WINDOW_SIZE)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        scene.update(0.20)
        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
