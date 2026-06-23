"""Unified shell adapter for the Logistic Regression Boundary Lab scene."""

from __future__ import annotations

import pygame
from logistic_regression_boundary_lab import LogisticRegressionBoundaryScene
from logistic_regression_boundary_lab.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

LOGISTIC_LESSON_ID = "error_logistic_boundary"
MOVE_BOUNDARY_TASK_ID = "move_decision_boundary"
COMPARE_POINTS_TASK_ID = "compare_confident_and_borderline_points"


class LogisticRegressionSceneAdapter:
    """Adapt the standalone Logistic Regression scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Logistic Regression scene."""
        self._context = context
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
            self._record_progress_from_event(event)
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

    def _record_progress_from_event(self, event: pygame.event.Event) -> None:
        """Complete lesson tasks from meaningful demo interactions."""
        if self._context.selected_lesson_id != LOGISTIC_LESSON_ID:
            return
        if event.type != pygame.KEYDOWN:
            return

        if event.key in {
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_e,
            pygame.K_q,
            pygame.K_LEFT,
            pygame.K_RIGHT,
        }:
            self._context.progress.complete_task(
                LOGISTIC_LESSON_ID,
                MOVE_BOUNDARY_TASK_ID,
            )
            self._mark_lesson_completed_if_ready()
        elif event.key in {pygame.K_SPACE, pygame.K_n}:
            self._context.progress.complete_task(
                LOGISTIC_LESSON_ID,
                COMPARE_POINTS_TASK_ID,
            )
            self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both tasks are done."""
        progress = self._context.progress.lessons.get(LOGISTIC_LESSON_ID)
        if progress is None:
            return

        required_tasks = {MOVE_BOUNDARY_TASK_ID, COMPARE_POINTS_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(LOGISTIC_LESSON_ID)


def create_logistic_regression_scene(context: AppContext) -> LogisticRegressionSceneAdapter:
    """Create the unified shell Logistic Regression Boundary Lab scene."""
    return LogisticRegressionSceneAdapter(context)
