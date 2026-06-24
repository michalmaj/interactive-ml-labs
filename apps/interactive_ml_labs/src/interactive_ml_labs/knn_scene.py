"""Unified shell adapter for the k-NN Vote Map scene."""

from __future__ import annotations

import pygame
from knn_vote_map import KNNVoteMapScene
from knn_vote_map.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

KNN_LESSON_ID = "distance_knn_vote"
CLASSIFY_QUERY_TASK_ID = "classify_query_point"
COMPARE_K_TASK_ID = "compare_k_values"
LEFT_MOUSE_BUTTON = 1


class KNNVoteMapSceneAdapter:
    """Adapt the standalone k-NN Vote Map scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped k-NN scene."""
        self._context = context
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = KNNVoteMapScene(
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
        """Complete guided lesson tasks from meaningful k-NN interactions."""
        if self._context.selected_lesson_id != KNN_LESSON_ID:
            return

        classified_query = (event.type == pygame.KEYDOWN and event.key == pygame.K_n) or (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == LEFT_MOUSE_BUTTON
        )
        if classified_query:
            self._context.progress.complete_task(KNN_LESSON_ID, CLASSIFY_QUERY_TASK_ID)
            self._mark_lesson_completed_if_ready()
        elif event.type == pygame.KEYDOWN and event.key in {pygame.K_UP, pygame.K_DOWN}:
            self._context.progress.complete_task(KNN_LESSON_ID, COMPARE_K_TASK_ID)
            self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both tasks are done."""
        progress = self._context.progress.lessons.get(KNN_LESSON_ID)
        if progress is None:
            return

        required_tasks = {CLASSIFY_QUERY_TASK_ID, COMPARE_K_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(KNN_LESSON_ID)


def create_knn_vote_map_scene(context: AppContext) -> KNNVoteMapSceneAdapter:
    """Create the unified shell k-NN Vote Map scene."""
    return KNNVoteMapSceneAdapter(context)
