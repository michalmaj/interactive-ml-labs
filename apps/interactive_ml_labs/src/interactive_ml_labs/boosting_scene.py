"""Unified shell adapter for the Boosting Mistake Lab scene."""

from __future__ import annotations

import pygame
from boosting_mistake_lab import BoostingMistakeScene

from interactive_ml_labs.display import BOOSTING_FIXED_SCENE_SIZE, Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BOOSTING_LESSON_ID = "error_boosting_mistakes"
ADVANCE_ROUNDS_TASK_ID = "advance_boosting_rounds"
CONNECT_WEIGHT_TASK_ID = "connect_weight_to_error"


class BoostingMistakeLabSceneAdapter:
    """Adapt the standalone Boosting Pygame scene to the shell scene contract."""

    fixed_scene_size: Size = BOOSTING_FIXED_SCENE_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Boosting scene."""
        self._context = context
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
            self._record_progress_from_event(event)
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

    def _record_progress_from_event(self, event: pygame.event.Event) -> None:
        """Complete lesson tasks from meaningful Boosting interactions."""
        if self._context.selected_lesson_id != BOOSTING_LESSON_ID:
            return
        if event.type != pygame.KEYDOWN:
            return

        if event.key in {
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_RIGHTBRACKET,
            pygame.K_LEFTBRACKET,
            pygame.K_EQUALS,
            pygame.K_MINUS,
        }:
            self._context.progress.complete_task(
                BOOSTING_LESSON_ID,
                ADVANCE_ROUNDS_TASK_ID,
            )
            self._mark_lesson_completed_if_ready()
        elif event.key == pygame.K_c:
            self._context.progress.complete_task(
                BOOSTING_LESSON_ID,
                CONNECT_WEIGHT_TASK_ID,
            )
            self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both tasks are done."""
        progress = self._context.progress.lessons.get(BOOSTING_LESSON_ID)
        if progress is None:
            return

        required_tasks = {ADVANCE_ROUNDS_TASK_ID, CONNECT_WEIGHT_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(BOOSTING_LESSON_ID)


def create_boosting_mistake_lab_scene(context: AppContext) -> BoostingMistakeLabSceneAdapter:
    """Create the unified shell Boosting Mistake Lab scene."""
    return BoostingMistakeLabSceneAdapter(context)
