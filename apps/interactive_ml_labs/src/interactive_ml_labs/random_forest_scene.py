"""Unified shell adapter for the Random Forest Bagging Lab scene."""

from __future__ import annotations

from typing import Final

import pygame
from random_forest_bagging_lab import RandomForestScene
from random_forest_bagging_lab.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

RANDOM_FOREST_LESSON_ID: Final[str] = "feature_decision_forest_vote"
COMPARE_FOREST_VOTE_TASK_ID: Final[str] = "compare_forest_vote"
INSPECT_FOREST_CONFIDENCE_TASK_ID: Final[str] = "inspect_forest_confidence"


class RandomForestSceneAdapter:
    """Adapt the standalone Random Forest scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Random Forest scene."""
        self._context = context
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = RandomForestScene(
            self._surface,
            present_frame=False,
            language=context.settings.language,
        )
        self._seen_tree_count_change = False
        self._seen_uncertainty_control = False

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event through the wrapped demo scene."""
        if not isinstance(event, pygame.event.Event):
            return SceneCommand.none()

        if self._scene.handle_event(event):
            self._record_lesson_progress(event)
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

    def _record_lesson_progress(self, event: pygame.event.Event) -> None:
        """Complete guided lesson tasks from meaningful forest interactions."""
        if (
            event.type != pygame.KEYDOWN
            or self._context.selected_lesson_id != RANDOM_FOREST_LESSON_ID
        ):
            return

        if event.key == pygame.K_r:
            self._seen_tree_count_change = False
            self._seen_uncertainty_control = False
            return

        if event.key in {pygame.K_UP, pygame.K_DOWN}:
            self._seen_tree_count_change = True
            self._context.progress.complete_task(
                RANDOM_FOREST_LESSON_ID,
                COMPARE_FOREST_VOTE_TASK_ID,
            )

        if event.key in {
            pygame.K_c,
            pygame.K_b,
            pygame.K_v,
            pygame.K_LEFT,
            pygame.K_RIGHT,
        } and (event.key != pygame.K_c or self._scene._confidence_view_enabled):
            self._seen_uncertainty_control = True

        if self._scene._confidence_view_enabled and self._seen_uncertainty_control:
            self._context.progress.complete_task(
                RANDOM_FOREST_LESSON_ID,
                INSPECT_FOREST_CONFIDENCE_TASK_ID,
            )

        self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both guided tasks are done."""
        progress = self._context.progress.lessons.get(RANDOM_FOREST_LESSON_ID)
        if progress is None:
            return

        required_tasks = {COMPARE_FOREST_VOTE_TASK_ID, INSPECT_FOREST_CONFIDENCE_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(RANDOM_FOREST_LESSON_ID)


def create_random_forest_scene(context: AppContext) -> RandomForestSceneAdapter:
    """Create the unified shell Random Forest Bagging Lab scene."""
    return RandomForestSceneAdapter(context)
