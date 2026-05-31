"""Pygame application loop for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from typing import Final

import pygame

from logistic_regression_boundary_lab.renderer import WINDOW_SIZE
from logistic_regression_boundary_lab.scene import LogisticRegressionBoundaryScene

FPS: Final[int] = 60


class LogisticRegressionPygameApp:
    """Small Pygame application showing logistic regression training."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs - Logistic Regression Boundary Lab")

        self._clock = pygame.time.Clock()
        self._scene = LogisticRegressionBoundaryScene(self._screen)
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
    app = LogisticRegressionPygameApp()
    app.run()
