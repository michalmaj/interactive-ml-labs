"""Unified shell adapter for the Boosting Mistake Lab scene."""

from __future__ import annotations

import pygame
from boosting_mistake_lab import BoostingMistakeScene

from interactive_ml_labs.display import BOOSTING_FIXED_SCENE_SIZE, Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext


class BoostingMistakeLabSceneAdapter:
    """Adapt the standalone Boosting Pygame scene to the shell scene contract."""

    fixed_scene_size: Size = BOOSTING_FIXED_SCENE_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Boosting scene."""
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = BoostingMistakeScene(
            self._surface,
            present_frame=False,
            language=context.settings.language,
        )

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event through the wrapped demo scene."""
        if not isinstance(event, pygame.event.Event):
            return SceneCommand.none()

        if self._scene.handle_event(event):
            return SceneCommand.none()

        return SceneCommand.pause()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Render the wrapped scene into the shell-provided surface."""
        if not isinstance(surface, pygame.Surface):
            return

        self._scene.render()
        surface.blit(self._surface, (0, 0))


def create_boosting_mistake_lab_scene(context: AppContext) -> BoostingMistakeLabSceneAdapter:
    """Create the unified shell Boosting Mistake Lab scene."""
    return BoostingMistakeLabSceneAdapter(context)
