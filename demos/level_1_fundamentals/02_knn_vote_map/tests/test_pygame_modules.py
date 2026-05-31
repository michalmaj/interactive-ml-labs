"""Smoke tests for k-NN Pygame-related modules."""

import knn_vote_map.pygame_app
import knn_vote_map.renderer
import pygame
from knn_vote_map import KNNVoteMapScene


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert knn_vote_map.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert knn_vote_map.renderer is not None


def test_knn_scene_handles_standalone_events(monkeypatch) -> None:
    """k-NN scene should handle events without owning the app loop."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.Surface((1100, 720))
        scene = KNNVoteMapScene(screen, present_frame=False)

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert not scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    finally:
        pygame.quit()


def test_knn_scene_can_render_without_presenting_frame(monkeypatch) -> None:
    """Embedded k-NN scenes should render to a surface without flipping display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.Surface((1100, 720))
        scene = KNNVoteMapScene(screen, present_frame=False)

        scene.render()

        assert screen.get_bounding_rect().width > 0
        assert screen.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_knn_scene_handles_mouse_query_after_render(monkeypatch) -> None:
    """Mouse clicks on the rendered map should be handled by the scene."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.Surface((1100, 720))
        scene = KNNVoteMapScene(screen, present_frame=False)
        scene.render()

        assert scene.handle_event(
            pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(390, 300)),
        )
    finally:
        pygame.quit()
