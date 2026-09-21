"""Progress summary helpers for the unified app shell."""

from __future__ import annotations

from dataclasses import dataclass

from interactive_ml_labs.manifest import LearningPathManifest, LessonManifest
from interactive_ml_labs.progress import AppProgress
from interactive_ml_labs.shell_catalog import ShellCatalog


@dataclass(frozen=True, slots=True)
class ShellProgressSummary:
    """Read-only progress facade used by the unified app shell."""

    catalog: ShellCatalog
    progress: AppProgress

    def is_lesson_completed(self, lesson_id: str) -> bool:
        """Return whether one lesson is completed."""
        lesson_progress = self.progress.lessons.get(lesson_id)
        return lesson_progress is not None and lesson_progress.completed

    def has_lesson_progress(self, lesson_id: str) -> bool:
        """Return whether one lesson has any saved progress."""
        lesson_progress = self.progress.lessons.get(lesson_id)
        return lesson_progress is not None and (
            lesson_progress.started
            or lesson_progress.theory_visited
            or bool(lesson_progress.completed_task_ids)
        )

    def completed_lesson_task_ids(self, lesson: LessonManifest) -> set[str]:
        """Return completed task ids for one lesson."""
        lesson_progress = self.progress.lessons.get(lesson.id)
        if lesson_progress is None:
            return set()

        return lesson_progress.completed_task_ids

    def next_learning_path_step(
        self,
    ) -> tuple[LearningPathManifest | None, LessonManifest | None]:
        """Return the next path and lesson to continue from the home screen."""
        for path in self.catalog.all_learning_paths():
            next_lesson = self.next_learning_path_lesson(path)
            if next_lesson is None:
                continue

            return path, next_lesson

        return None, None

    def next_learning_path_lesson(
        self,
        path: LearningPathManifest,
    ) -> LessonManifest | None:
        """Return the first incomplete lesson in one learning path."""
        for lesson_id in path.lesson_ids:
            if not self.is_lesson_completed(lesson_id):
                return self.catalog.lesson(lesson_id)

        return None

    def next_learning_path_lesson_index(self, path: LearningPathManifest) -> int:
        """Return the first incomplete lesson index in one learning path."""
        for index, lesson_id in enumerate(path.lesson_ids):
            if not self.is_lesson_completed(lesson_id):
                return index

        return 0

    def completed_learning_path_lesson_count(self, path: LearningPathManifest) -> int:
        """Count completed lessons in one learning path."""
        return sum(
            1 for lesson_id in path.lesson_ids if self.is_lesson_completed(lesson_id)
        )

    def completed_learning_path_task_count(self, path: LearningPathManifest) -> int:
        """Count completed tasks across one learning path."""
        return sum(
            len(self.completed_lesson_task_ids(self.catalog.lesson(lesson_id)))
            for lesson_id in path.lesson_ids
        )

    def learning_path_task_count(self, path: LearningPathManifest) -> int:
        """Count all tasks across one learning path."""
        return sum(len(self.catalog.lesson(lesson_id).tasks) for lesson_id in path.lesson_ids)

    def visited_learning_path_theory_count(self, path: LearningPathManifest) -> int:
        """Count lessons with visited theory in one learning path."""
        return sum(
            1
            for lesson_id in path.lesson_ids
            if (
                (lesson_progress := self.progress.lessons.get(lesson_id)) is not None
                and lesson_progress.theory_visited
            )
        )

    def unlocked_learning_path_badge_count(self, path: LearningPathManifest) -> int:
        """Count unlocked badges across one learning path."""
        return sum(
            1
            for lesson_id in path.lesson_ids
            if self.catalog.lesson(lesson_id).completion_badge is not None
            and self.is_lesson_completed(lesson_id)
        )

    def learning_path_badge_count(self, path: LearningPathManifest) -> int:
        """Count all badges available in one learning path."""
        return sum(
            1
            for lesson_id in path.lesson_ids
            if self.catalog.lesson(lesson_id).completion_badge is not None
        )

    def started_learning_path_lesson_count(self, path: LearningPathManifest) -> int:
        """Count started lessons in one learning path."""
        return sum(
            1
            for lesson_id in path.lesson_ids
            if (
                (lesson_progress := self.progress.lessons.get(lesson_id)) is not None
                and lesson_progress.started
            )
        )
