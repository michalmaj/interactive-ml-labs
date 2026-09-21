"""View-data builders for guided learning paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from interactive_ml_labs.manifest import LearningPathManifest, LessonManifest
from interactive_ml_labs.screens.badge_screen import BadgeGalleryPath
from interactive_ml_labs.screens.learning_path_screen import (
    LearningPathDetails,
    LearningPathMetric,
)
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary
from interactive_ml_labs.shell_types import BadgeItem

TextSelector = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class ShellLearningPathViewData:
    """Build render-ready data for guided learning path screens."""

    catalog: ShellCatalog
    progress: ShellProgressSummary
    language: str
    text: TextSelector

    def learning_path_details(self, path: LearningPathManifest) -> LearningPathDetails:
        """Return render-ready details for one learning path."""
        lesson_count = len(path.lesson_ids)
        lesson_count_label = self.text(
            f"{lesson_count} lessons",
            f"{lesson_count} lekcje" if lesson_count < 5 else f"{lesson_count} lekcji",
        )
        return LearningPathDetails(
            title=path.title.for_language(self.language),
            summary=path.summary.for_language(self.language),
            lesson_count_label=lesson_count_label,
            progress_metrics=self.learning_path_metrics(path),
            status_label=self.learning_path_status_label(path),
            next_action_label=self.learning_path_next_action_label(path),
            badge_labels=self.learning_path_badge_labels(path),
            course_map_heading=self.text("Course map", "Mapa kursu"),
            lesson_labels=[
                self.learning_path_lesson_map_label(self.catalog.lesson(lesson_id), index)
                for index, lesson_id in enumerate(path.lesson_ids, start=1)
            ],
        )

    def learning_path_metrics(self, path: LearningPathManifest) -> list[LearningPathMetric]:
        """Return localized progress labels and counts for one learning path."""
        return [
            LearningPathMetric(
                label=self.learning_path_progress_label(path),
                completed_count=self.progress.completed_learning_path_lesson_count(path),
                total_count=len(path.lesson_ids),
            ),
            LearningPathMetric(
                label=self.learning_path_task_progress_label(path),
                completed_count=self.progress.completed_learning_path_task_count(path),
                total_count=self.progress.learning_path_task_count(path),
            ),
            LearningPathMetric(
                label=self.learning_path_theory_progress_label(path),
                completed_count=self.progress.visited_learning_path_theory_count(path),
                total_count=len(path.lesson_ids),
            ),
            LearningPathMetric(
                label=self.learning_path_badge_progress_label(path),
                completed_count=self.progress.unlocked_learning_path_badge_count(path),
                total_count=self.progress.learning_path_badge_count(path),
            ),
        ]

    def learning_path_progress_label(self, path: LearningPathManifest) -> str:
        """Return a localized completion summary for one learning path."""
        completed_count = self.progress.completed_learning_path_lesson_count(path)
        total_count = len(path.lesson_ids)
        return self.text(
            f"Lessons: {completed_count}/{total_count} completed",
            f"Lekcje: {completed_count}/{total_count} ukończone",
        )

    def learning_path_task_progress_label(self, path: LearningPathManifest) -> str:
        """Return a localized task completion summary for one learning path."""
        completed_count = self.progress.completed_learning_path_task_count(path)
        total_count = self.progress.learning_path_task_count(path)
        return self.text(
            f"Tasks: {completed_count}/{total_count} completed",
            f"Zadania: {completed_count}/{total_count} ukończone",
        )

    def learning_path_theory_progress_label(self, path: LearningPathManifest) -> str:
        """Return a localized theory visit summary for one learning path."""
        visited_count = self.progress.visited_learning_path_theory_count(path)
        total_count = len(path.lesson_ids)
        return self.text(
            f"Theory: {visited_count}/{total_count} visited",
            f"Teoria: {visited_count}/{total_count} przeczytana",
        )

    def learning_path_badge_progress_label(self, path: LearningPathManifest) -> str:
        """Return a localized badge completion summary for one learning path."""
        unlocked_count = self.progress.unlocked_learning_path_badge_count(path)
        total_count = self.progress.learning_path_badge_count(path)
        return self.text(
            f"Badges: {unlocked_count}/{total_count} unlocked",
            f"Odznaki: {unlocked_count}/{total_count} zdobyte",
        )

    def learning_path_badge_labels(self, path: LearningPathManifest) -> list[str]:
        """Return badge labels with completion markers for one learning path."""
        labels: list[str] = []
        for badge in self.learning_path_badge_items(path):
            marker = "[x]" if badge.unlocked else "[ ]"
            labels.append(f"{marker} {badge.label}")

        return labels

    def learning_path_badge_items(self, path: LearningPathManifest) -> list[BadgeItem]:
        """Return badge labels and unlock state for one learning path."""
        badges: list[BadgeItem] = []
        for lesson_id in path.lesson_ids:
            lesson = self.catalog.lesson(lesson_id)
            if lesson.completion_badge is None:
                continue

            badges.append(
                BadgeItem(
                    label=lesson.completion_badge.for_language(self.language),
                    unlocked=self.progress.is_lesson_completed(lesson_id),
                ),
            )

        return badges

    def badge_gallery_paths(self) -> list[BadgeGalleryPath]:
        """Return render-ready badge gallery data for all learning paths."""
        return [
            BadgeGalleryPath(
                title=path.title.for_language(self.language),
                progress_label=self.learning_path_badge_progress_label(path),
                badges=self.learning_path_badge_items(path),
            )
            for path in self.catalog.all_learning_paths()
        ]

    def learning_path_lesson_map_label(self, lesson: LessonManifest, index: int) -> str:
        """Return one compact lesson row for a learning path course map."""
        completed_tasks, total_tasks = self.lesson_task_progress_counts(lesson)
        title = lesson.title.for_language(self.language)
        status = self.lesson_progress_label(lesson)
        return self.text(
            f"{index}. {title} - {status}; tasks {completed_tasks}/{total_tasks}",
            f"{index}. {title} - {status}; zadania {completed_tasks}/{total_tasks}",
        )

    def learning_path_status_label(self, path: LearningPathManifest) -> str:
        """Return a localized status label for one learning path."""
        started_count = self.progress.started_learning_path_lesson_count(path)
        completed_count = self.progress.completed_learning_path_lesson_count(path)
        total_count = len(path.lesson_ids)

        if total_count > 0 and completed_count == total_count:
            return self.text("Path completed", "Ścieżka ukończona")
        if started_count > 0:
            return self.text("In progress", "W trakcie")

        return self.text("Not started", "Nie rozpoczęto")

    def learning_path_next_action_label(self, path: LearningPathManifest) -> str:
        """Return a localized next action hint for one learning path."""
        next_lesson = self.progress.next_learning_path_lesson(path)
        if next_lesson is None:
            return self.text(
                "Next action: review completed lessons",
                "Następny krok: powtórz ukończone lekcje",
            )

        title = next_lesson.title.for_language(self.language)
        if self.progress.has_lesson_progress(next_lesson.id):
            return self.text(
                f"Next action: continue {title}",
                f"Następny krok: kontynuuj {title}",
            )

        return self.text(f"Next action: start {title}", f"Następny krok: zacznij {title}")

    def lesson_progress_label(self, lesson: LessonManifest) -> str:
        """Return a short localized progress label for one lesson."""
        lesson_progress = self.progress.progress.lessons.get(lesson.id)
        if lesson_progress is None or not lesson_progress.started:
            return self.text("Not started", "Nie rozpoczęto")
        if lesson_progress.completed:
            return self.text("Completed", "Ukończona")
        if lesson_progress.theory_visited:
            return self.text("Theory visited", "Teoria przeczytana")

        return self.text("Started", "Rozpoczęta")

    def lesson_task_progress_counts(self, lesson: LessonManifest) -> tuple[int, int]:
        """Return completed and total task counts for one lesson."""
        return len(self.progress.completed_lesson_task_ids(lesson)), len(lesson.tasks)
