"""Unified shell adapter for the Gradient Descent Playground scene."""

from __future__ import annotations

import pygame
from gradient_descent_playground import GradientDescentScene
from gradient_descent_playground.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

GRADIENT_LESSON_ID = "error_gradient_descent"
LEARNING_RATE_TASK_ID = "find_stable_learning_rate"
LOSS_DROP_TASK_ID = "observe_loss_drop"


class GradientDescentSceneAdapter:
    """Adapt the standalone Gradient Descent scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Gradient Descent scene."""
        self._context = context
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = GradientDescentScene(
            self._surface,
            present_frame=False,
            language=context.settings.language,
        )

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event through the wrapped demo scene."""
        if not isinstance(event, pygame.event.Event):
            return SceneCommand.none()

        keep_running = self._scene.handle_event(event)
        if keep_running:
            self._record_progress_from_event(event)
            self._record_progress_from_loss_history()
            return SceneCommand.none()

        return SceneCommand.pause()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        self._scene.update(dt)
        self._record_progress_from_loss_history()
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Render the wrapped scene into the shell-provided surface."""
        if not isinstance(surface, pygame.Surface):
            return

        self._scene.render()
        surface.blit(self._surface, (0, 0))

    def _record_progress_from_event(self, event: pygame.event.Event) -> None:
        """Complete event-driven lesson tasks."""
        if self._context.selected_lesson_id != GRADIENT_LESSON_ID:
            return

        if event.type == pygame.KEYDOWN and event.key in {pygame.K_UP, pygame.K_DOWN}:
            self._context.progress.complete_task(GRADIENT_LESSON_ID, LEARNING_RATE_TASK_ID)
            self._mark_lesson_completed_if_ready()

    def _record_progress_from_loss_history(self) -> None:
        """Complete metric-driven lesson tasks."""
        if self._context.selected_lesson_id != GRADIENT_LESSON_ID:
            return

        loss_history = self._scene._snapshot.visual_state.get("loss_history", ())
        if len(loss_history) < 3:
            return

        initial_loss = float(loss_history[0])
        current_loss = float(loss_history[-1])
        if current_loss < initial_loss:
            self._context.progress.complete_task(GRADIENT_LESSON_ID, LOSS_DROP_TASK_ID)
            self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both initial tasks are done."""
        progress = self._context.progress.lessons.get(GRADIENT_LESSON_ID)
        if progress is None:
            return

        required_tasks = {LEARNING_RATE_TASK_ID, LOSS_DROP_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(GRADIENT_LESSON_ID)


def create_gradient_descent_scene(context: AppContext) -> GradientDescentSceneAdapter:
    """Create the unified shell Gradient Descent Playground scene."""
    return GradientDescentSceneAdapter(context)
