"""Pygame application for the Boosting Mistake Lab demo."""

from __future__ import annotations

from typing import Final

import pygame

from boosting_mistake_lab.renderer import WINDOW_SIZE
from boosting_mistake_lab.scene import BoostingMistakeScene

FPS: Final[int] = 30


class BoostingPygameApp:
    """Standalone Pygame app wrapper for the Boosting Mistake Lab scene."""

    def __init__(self) -> None:
        """Initialize the Pygame app."""
        pygame.init()
        pygame.display.set_caption("Boosting Mistake Lab")

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        self._clock = pygame.time.Clock()
        self._scene = BoostingMistakeScene(self._screen)
        self._running = True

    def run(self) -> None:
        """Run the main event loop."""
        try:
            while self._running:
                self._handle_events()
                self._scene.render()
                self._clock.tick(FPS)
        finally:
            pygame.quit()

    def _handle_events(self) -> None:
        """Handle all pending Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or not self._scene.handle_event(event):
                self._running = False


def main() -> None:
    """Run the Boosting Mistake Lab Pygame app."""
    app = BoostingPygameApp()
    app.run()
