"""Smoke tests for the Boosting Mistake Lab Pygame UI modules."""

import pygame
from boosting_mistake_lab import BoostingMistakeScene
from boosting_mistake_lab.pygame_app import BoostingPygameApp
from boosting_mistake_lab.renderer import BoostingRenderer, BoostingRenderState


def test_pygame_ui_modules_are_importable() -> None:
    """Pygame UI modules should be importable."""

    assert BoostingMistakeScene is not None
    assert BoostingPygameApp is not None
    assert BoostingRenderer is not None
    assert BoostingRenderState is not None


def test_boosting_scene_handles_standalone_events(monkeypatch) -> None:
    """Boosting scene should handle events without owning the app loop."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        screen = pygame.display.set_mode((1320, 780))
        scene = BoostingMistakeScene(screen)

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        assert not scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
    finally:
        pygame.quit()
