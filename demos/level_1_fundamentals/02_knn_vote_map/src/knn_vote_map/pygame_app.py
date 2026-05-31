"""Pygame application loop for the k-NN Vote Map demo."""

from __future__ import annotations

from typing import Final

import pygame

from knn_vote_map.renderer import WINDOW_SIZE
from knn_vote_map.scene import KNNVoteMapScene

FPS: Final[int] = 60


class KNNVoteMapPygameApp:
    """Small Pygame application showing k-NN voting."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs - k-NN Vote Map")

        self._clock = pygame.time.Clock()
        self._scene = KNNVoteMapScene(self._screen)
        self._should_quit = False

    def run(self) -> None:
        """Run the Pygame event loop."""
        while not self._should_quit:
            dt = self._clock.tick(FPS) / 1000.0

            self._handle_events()
            self._scene.update(dt)
            self._scene.render()

        pygame.quit()

    def _handle_events(self) -> None:
        """Handle Pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or not self._scene.handle_event(event):
                self._should_quit = True


def main() -> None:
    """Run the Pygame visualization."""
    app = KNNVoteMapPygameApp()
    app.run()
