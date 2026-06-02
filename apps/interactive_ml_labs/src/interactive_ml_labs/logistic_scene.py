"""Unified shell adapter for the Logistic Regression Boundary Lab scene."""

from __future__ import annotations

import pygame
from logistic_regression_boundary_lab import LogisticRegressionBoundaryScene
from logistic_regression_boundary_lab.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext


class LogisticRegressionSceneAdapter:
    """Adapt the standalone Logistic Regression scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Logistic Regression scene."""
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = LogisticRegressionBoundaryScene(
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


def create_logistic_regression_scene(context: AppContext) -> LogisticRegressionSceneAdapter:
    """Create the unified shell Logistic Regression Boundary Lab scene."""
    return LogisticRegressionSceneAdapter(context)
