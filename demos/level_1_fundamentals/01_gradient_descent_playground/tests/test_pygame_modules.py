"""Smoke tests for Pygame-related modules."""

import gradient_descent_playground.pygame_app
import gradient_descent_playground.renderer
import pygame
from gradient_descent_playground import GradientDescentScene


def test_import_pygame_app_module() -> None:
    """The Pygame app module should be importable."""

    assert gradient_descent_playground.pygame_app is not None


def test_import_renderer_module() -> None:
    """The renderer module should be importable."""

    assert gradient_descent_playground.renderer is not None


def test_gradient_scene_handles_standalone_events(monkeypatch) -> None:
    """Gradient scene should handle events without owning the app loop."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.Surface((1100, 720))
        scene = GradientDescentScene(screen, present_frame=False)

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert not scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    finally:
        pygame.quit()


def test_gradient_scene_can_render_without_presenting_frame(monkeypatch) -> None:
    """Embedded Gradient scenes should render to a surface without flipping display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.Surface((1100, 720))
        scene = GradientDescentScene(screen, present_frame=False)

        scene.render()

        assert screen.get_bounding_rect().width > 0
        assert screen.get_bounding_rect().height > 0
    finally:
        pygame.quit()
