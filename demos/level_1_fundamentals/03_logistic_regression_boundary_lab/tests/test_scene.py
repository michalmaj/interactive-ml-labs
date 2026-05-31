"""Tests for the reusable Logistic Regression Boundary Lab Pygame scene."""

from __future__ import annotations

import pygame
from logistic_regression_boundary_lab.renderer import WINDOW_SIZE
from logistic_regression_boundary_lab.scene import LogisticRegressionBoundaryScene


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
        scene = LogisticRegressionBoundaryScene(surface, present_frame=False)
        scene.render()

        assert flip_calls == 0
    finally:
        pygame.quit()


def test_scene_handles_step_and_reset_keys(monkeypatch) -> None:
    """The reusable scene should keep the existing keyboard controls working."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = LogisticRegressionBoundaryScene(surface, present_frame=False)
        initial_iteration = scene._snapshot.iteration

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_n}))
        assert scene._snapshot.iteration == initial_iteration + 1

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r}))
        assert scene._snapshot.iteration == 0
    finally:
        pygame.quit()


def test_scene_escape_requests_standalone_close(monkeypatch) -> None:
    """Esc should still request closing the standalone Pygame app."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = LogisticRegressionBoundaryScene(surface, present_frame=False)

        should_continue = scene.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}),
        )

        assert should_continue is False
    finally:
        pygame.quit()
