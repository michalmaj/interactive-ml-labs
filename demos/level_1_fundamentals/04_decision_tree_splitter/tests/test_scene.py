"""Tests for the reusable Decision Tree Splitter Pygame scene."""

from __future__ import annotations

import pygame
from decision_tree_splitter.renderer import WINDOW_SIZE
from decision_tree_splitter.scene import (
    MODE_AUTO_TREE,
    MODE_MANUAL_SPLIT,
    DecisionTreeSplitterScene,
)


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
        scene = DecisionTreeSplitterScene(surface, present_frame=False)
        scene.render()

        assert flip_calls == 0
    finally:
        pygame.quit()


def test_scene_handles_mode_depth_and_reset_keys(monkeypatch) -> None:
    """The reusable scene should keep the existing keyboard controls working."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = DecisionTreeSplitterScene(surface, present_frame=False)
        initial_depth = scene._max_depth

        assert scene._mode == MODE_AUTO_TREE
        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_m}))
        assert scene._mode == MODE_MANUAL_SPLIT

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP}))
        assert scene._max_depth == initial_depth + 1

        assert scene.handle_event(pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_r}))
        assert scene._mode == MODE_MANUAL_SPLIT
        assert scene._tree_snapshot.status == "fitted"
        assert scene._manual_snapshot is not None
    finally:
        pygame.quit()


def test_scene_escape_requests_standalone_close(monkeypatch) -> None:
    """Esc should still request closing the standalone Pygame app."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    surface = pygame.Surface(WINDOW_SIZE)

    try:
        scene = DecisionTreeSplitterScene(surface, present_frame=False)

        should_continue = scene.handle_event(
            pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE}),
        )

        assert should_continue is False
    finally:
        pygame.quit()
