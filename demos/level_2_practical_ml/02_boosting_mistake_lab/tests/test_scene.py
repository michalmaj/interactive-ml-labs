"""Tests for the reusable Boosting Mistake Lab Pygame scene."""

import pygame
from boosting_mistake_lab import BoostingMistakeScene


def test_boosting_scene_uses_brackets_for_round_count(monkeypatch) -> None:
    """Round shortcuts should avoid PageUp/PageDown."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = BoostingMistakeScene(pygame.Surface((1320, 780)), present_frame=False)
        initial_round_count = scene._state.round_count

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET))
        assert scene._state.round_count == initial_round_count + 1

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET))
        assert scene._state.round_count == initial_round_count

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PAGEUP))
        assert scene._state.round_count == initial_round_count
    finally:
        pygame.quit()
