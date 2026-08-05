"""Unified shell adapter for the Decision Tree Splitter scene."""

from __future__ import annotations

from typing import Final

import pygame
from decision_tree_splitter import DecisionTreeSplitterScene
from decision_tree_splitter.renderer import WINDOW_SIZE

from interactive_ml_labs.display import Size
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

DECISION_TREE_LESSON_ID: Final[str] = "feature_decision_tree_split"
MOVE_TREE_SPLIT_TASK_ID: Final[str] = "move_tree_split"
INSPECT_FIRST_SPLIT_TASK_ID: Final[str] = "inspect_first_split"


class DecisionTreeSceneAdapter:
    """Adapt the standalone Decision Tree scene to the shell scene contract."""

    fixed_scene_size: Size = WINDOW_SIZE

    def __init__(self, context: AppContext) -> None:
        """Create the wrapped Decision Tree scene."""
        self._context = context
        self._surface = pygame.Surface(self.fixed_scene_size)
        self._scene = DecisionTreeSplitterScene(
            self._surface,
            present_frame=False,
            language=context.settings.language,
        )
        self._seen_manual_split_controls = False
        self._seen_tree_structure_controls = False

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
        """Complete guided lesson tasks from meaningful tree interactions."""
        if (
            event.type != pygame.KEYDOWN
            or self._context.selected_lesson_id != DECISION_TREE_LESSON_ID
        ):
            return

        if event.key == pygame.K_r:
            self._seen_manual_split_controls = False
            self._seen_tree_structure_controls = False
            return

        if event.key in {pygame.K_m, pygame.K_f, pygame.K_q, pygame.K_e}:
            self._seen_manual_split_controls = True
            if self._scene._manual_snapshot is not None:
                self._context.progress.complete_task(
                    DECISION_TREE_LESSON_ID,
                    MOVE_TREE_SPLIT_TASK_ID,
                )

        if event.key in {pygame.K_g, pygame.K_UP, pygame.K_DOWN}:
            self._seen_tree_structure_controls = True
            self._context.progress.complete_task(
                DECISION_TREE_LESSON_ID,
                INSPECT_FIRST_SPLIT_TASK_ID,
            )

        self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both guided tasks are done."""
        progress = self._context.progress.lessons.get(DECISION_TREE_LESSON_ID)
        if progress is None:
            return

        required_tasks = {MOVE_TREE_SPLIT_TASK_ID, INSPECT_FIRST_SPLIT_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(DECISION_TREE_LESSON_ID)


def create_decision_tree_scene(context: AppContext) -> DecisionTreeSceneAdapter:
    """Create the unified shell Decision Tree Splitter scene."""
    return DecisionTreeSceneAdapter(context)
