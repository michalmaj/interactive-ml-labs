"""View-data builders for the shell progress report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from interactive_ml_labs.manifest import LearningPathManifest, LessonManifest
from interactive_ml_labs.screens.progress_report_screen import (
    ProgressReportDetails,
    ProgressReportMetric,
    ProgressReportPath,
)
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_learning_path_view_data import ShellLearningPathViewData
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary
from interactive_ml_labs.shell_types import BadgeItem

TextSelector = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class ShellProgressReportViewData:
    """Build render-ready data for the learning progress report."""

    catalog: ShellCatalog
    progress: ShellProgressSummary
    learning_paths: ShellLearningPathViewData
    language: str
    text: TextSelector

    def progress_report_details(self) -> ProgressReportDetails:
        """Return render-ready progress report details."""
        badges = self._badge_items()
        return ProgressReportDetails(
            title=self.text("Progress report", "Raport postępu"),
            subtitle=self.text(
                "A compact view of what you have completed and what you can already explain.",
                "Krótki przegląd tego, co masz ukończone i co umiesz już wyjaśnić.",
            ),
            metrics=self._global_metrics(),
            explain_heading=self.text("You can already explain", "Umiesz już wyjaśnić"),
            explain_lines=self._explain_lines(),
            paths_heading=self.text("Guided paths", "Prowadzone ścieżki"),
            paths=self._path_summaries(),
            badges_heading=self.text("Badges", "Odznaki"),
            badge_summary=self._badge_summary(badges),
            badges=badges,
        )

    def _global_metrics(self) -> list[ProgressReportMetric]:
        """Return top-level progress metrics across the course."""
        completed_paths = sum(
            1 for path in self.catalog.all_learning_paths() if self._is_path_completed(path)
        )
        total_paths = self.catalog.learning_path_count()
        completed_lessons = sum(
            self.progress.completed_learning_path_lesson_count(path)
            for path in self.catalog.all_learning_paths()
        )
        total_lessons = sum(len(path.lesson_ids) for path in self.catalog.all_learning_paths())
        completed_tasks = sum(
            self.progress.completed_learning_path_task_count(path)
            for path in self.catalog.all_learning_paths()
        )
        total_tasks = sum(
            self.progress.learning_path_task_count(path)
            for path in self.catalog.all_learning_paths()
        )
        unlocked_badges = sum(
            self.progress.unlocked_learning_path_badge_count(path)
            for path in self.catalog.all_learning_paths()
        )
        total_badges = sum(
            self.progress.learning_path_badge_count(path)
            for path in self.catalog.all_learning_paths()
        )

        return [
            ProgressReportMetric(
                label=self.text(
                    f"Paths completed: {completed_paths}/{total_paths}",
                    f"Ukończone ścieżki: {completed_paths}/{total_paths}",
                ),
                completed_count=completed_paths,
                total_count=total_paths,
            ),
            ProgressReportMetric(
                label=self.text(
                    f"Lessons completed: {completed_lessons}/{total_lessons}",
                    f"Ukończone lekcje: {completed_lessons}/{total_lessons}",
                ),
                completed_count=completed_lessons,
                total_count=total_lessons,
            ),
            ProgressReportMetric(
                label=self.text(
                    f"Tasks completed: {completed_tasks}/{total_tasks}",
                    f"Ukończone zadania: {completed_tasks}/{total_tasks}",
                ),
                completed_count=completed_tasks,
                total_count=total_tasks,
            ),
            ProgressReportMetric(
                label=self.text(
                    f"Badges unlocked: {unlocked_badges}/{total_badges}",
                    f"Zdobyte odznaki: {unlocked_badges}/{total_badges}",
                ),
                completed_count=unlocked_badges,
                total_count=total_badges,
            ),
        ]

    def _explain_lines(self) -> list[str]:
        """Return short learning claims unlocked by completed lessons."""
        completed_lessons = [
            self.catalog.lesson(lesson_id)
            for path in self.catalog.all_learning_paths()
            for lesson_id in path.lesson_ids
            if self.progress.is_lesson_completed(lesson_id)
        ]
        if not completed_lessons:
            return [
                self.text(
                    "Start a guided lesson to unlock your first explanation.",
                    "Rozpocznij prowadzoną lekcję, żeby odblokować pierwsze podsumowanie.",
                ),
            ]

        return [self._lesson_explanation_line(lesson) for lesson in completed_lessons[:6]]

    def _lesson_explanation_line(self, lesson: LessonManifest) -> str:
        """Return one natural explanation line for a completed lesson."""
        title = lesson.title.for_language(self.language)
        goal = lesson.learning_goal.for_language(self.language)
        return self.text(
            f"{title}: {goal}",
            f"{title}: {goal}",
        )

    def _path_summaries(self) -> list[ProgressReportPath]:
        """Return progress summaries for every guided learning path."""
        paths: list[ProgressReportPath] = []
        for path in self.catalog.all_learning_paths():
            lesson_count = self.progress.completed_learning_path_lesson_count(path)
            total_lessons = len(path.lesson_ids)
            task_count = self.progress.completed_learning_path_task_count(path)
            total_tasks = self.progress.learning_path_task_count(path)
            badge_count = self.progress.unlocked_learning_path_badge_count(path)
            total_badges = self.progress.learning_path_badge_count(path)
            paths.append(
                ProgressReportPath(
                    title=path.title.for_language(self.language),
                    status_label=self.learning_paths.learning_path_status_label(path),
                    lesson_label=self.text(
                        f"Lessons {lesson_count}/{total_lessons}",
                        f"Lekcje {lesson_count}/{total_lessons}",
                    ),
                    task_label=self.text(
                        f"Tasks {task_count}/{total_tasks}",
                        f"Zadania {task_count}/{total_tasks}",
                    ),
                    badge_label=self.text(
                        f"Badges {badge_count}/{total_badges}",
                        f"Odznaki {badge_count}/{total_badges}",
                    ),
                    completed_count=lesson_count,
                    total_count=total_lessons,
                ),
            )

        return paths

    def _badge_items(self) -> list[BadgeItem]:
        """Return all badges with their unlock state."""
        badges: list[BadgeItem] = []
        for path in self.catalog.all_learning_paths():
            badges.extend(self.learning_paths.learning_path_badge_items(path))

        return badges

    def _badge_summary(self, badges: list[BadgeItem]) -> str:
        """Return a localized badge summary label."""
        unlocked_count = sum(1 for badge in badges if badge.unlocked)
        total_count = len(badges)
        return self.text(
            f"Unlocked badges: {unlocked_count}/{total_count}",
            f"Zdobyte odznaki: {unlocked_count}/{total_count}",
        )

    def _is_path_completed(self, path: LearningPathManifest) -> bool:
        """Return whether every lesson in a path is complete."""
        return len(path.lesson_ids) > 0 and all(
            self.progress.is_lesson_completed(lesson_id) for lesson_id in path.lesson_ids
        )
