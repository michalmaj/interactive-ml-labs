"""Pygame application loop for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from typing import Final

import pygame

from random_forest_bagging_lab.renderer import WINDOW_SIZE
from random_forest_bagging_lab.scene import RandomForestScene

FPS: Final[int] = 30


class RandomForestPygameApp:
    """Small Pygame app comparing one tree with a random forest."""

    def __init__(self) -> None:
        """Initialize app state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs - Random Forest Bagging Lab")

        self._clock = pygame.time.Clock()
        self._scene = RandomForestScene(self._screen)
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
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or not self._scene.handle_event(event):
                self._should_quit = True


def main() -> None:
    """Run the Pygame visualization."""
    app = RandomForestPygameApp()
    app.run()
