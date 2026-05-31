"""Tests for the reusable Random Forest Bagging Lab Pygame scene."""

from __future__ import annotations

import pygame
from random_forest_bagging_lab.renderer import WINDOW_SIZE
from random_forest_bagging_lab.scene import RandomForestScene


def test_scene_renders_without_presenting_frame(monkeypatch) -> None:
    """The reusable scene should render to a surface without flipping the display."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)
    flip_calls = 0

    def record_flip() -> None:
        nonlocal flip_calls
        flip_calls += 1

    monkeypatch.setattr(pygame.display, "flip", record_flip)

    try:
        scene = RandomForestScene(surface, present_frame=False)
        scene.render()

        assert flip_calls == 0
    finally:
        pygame.quit()


def test_scene_handles_parameter_and_reset_keys(monkeypatch) -> None:
    """The reusable scene should keep the existing keyboard controls working."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = RandomForestScene(surface, present_frame=False)
        initial_tree_count = scene._tree_count

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP}))
        assert scene._tree_count == initial_tree_count + 4

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_c}))
        assert scene._confidence_view_enabled is True

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r}))
        assert scene._baseline_snapshot.status == "fitted"
        assert scene._forest_snapshot.status == "fitted"
    finally:
        pygame.quit()


def test_scene_escape_requests_standalone_close(monkeypatch) -> None:
    """Esc should still request closing the standalone Pygame app."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = RandomForestScene(surface, present_frame=False)

        should_continue = scene.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}),
        )

        assert should_continue is False
    finally:
        pygame.quit()
