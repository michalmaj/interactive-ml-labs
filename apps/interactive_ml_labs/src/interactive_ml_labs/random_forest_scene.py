"""Unified shell adapter for the Random Forest Bagging Lab scene."""

from __future__ import annotations

import pygame
from random_forest_bagging_lab import RandomForestScene
from random_forest_bagging_lab.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext


class RandomForestSceneAdapter:
    """Adapt the standalone Random Forest scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Random Forest scene."""
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = RandomForestScene(
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
        self._scene.update(dt)
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Render the wrapped scene into the shell-provided surface."""
        if not isinstance(surface, pygame.Surface):
            return

        self._scene.render()
        surface.blit(self._surface, (0, 0))


def create_random_forest_scene(context: AppContext) -> RandomForestSceneAdapter:
    """Create the unified shell Random Forest Bagging Lab scene."""
    return RandomForestSceneAdapter(context)
