"""View-data builders for the home learning progress snapshot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary

TextSelector = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class HomeProgressMetric:
    """One home-screen learning progress metric."""

    label: str
    completed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class HomeProgressDetails:
    """Renderable data for the home learning progress panel."""

    metrics: list[HomeProgressMetric]
    next_action_label: str

    @property
    def lines(self) -> list[str]:
        """Return compact text lines for tests and diagnostics."""
        return [metric.label for metric in self.metrics] + [self.next_action_label]


@dataclass(frozen=True, slots=True)
class ShellHomeProgressViewData:
    """Build render-ready data for the home progress snapshot."""

    catalog: ShellCatalog
    progress: ShellProgressSummary
    language: str
    text: TextSelector

    def home_progress_details(self) -> HomeProgressDetails:
        """Return render-ready home progress details."""
        return HomeProgressDetails(
            metrics=self.home_progress_metrics(),
            next_action_label=self.home_next_action_label(),
        )

    def home_progress_metrics(self) -> list[HomeProgressMetric]:
        """Return localized progress labels and counts for the home snapshot."""
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
            HomeProgressMetric(
                label=self.text(
                    f"Lessons: {completed_lessons}/{total_lessons} completed",
                    f"Lekcje: {completed_lessons}/{total_lessons} ukończone",
                ),
                completed_count=completed_lessons,
                total_count=total_lessons,
            ),
            HomeProgressMetric(
                label=self.text(
                    f"Tasks: {completed_tasks}/{total_tasks} completed",
                    f"Zadania: {completed_tasks}/{total_tasks} ukończone",
                ),
                completed_count=completed_tasks,
                total_count=total_tasks,
            ),
            HomeProgressMetric(
                label=self.text(
                    f"Badges: {unlocked_badges}/{total_badges} unlocked",
                    f"Odznaki: {unlocked_badges}/{total_badges} zdobyte",
                ),
                completed_count=unlocked_badges,
                total_count=total_badges,
            ),
        ]

    def home_next_action_label(self) -> str:
        """Return a localized next learning action across all paths."""
        next_path, next_lesson = self.progress.next_learning_path_step()
        if next_path is None or next_lesson is None:
            return self.text(
                "All guided paths completed",
                "Wszystkie ścieżki ukończone",
            )

        lesson_title = next_lesson.title.for_language(self.language)
        path_title = next_path.title.for_language(self.language)
        if self.progress.has_lesson_progress(next_lesson.id):
            return self.text(
                f"Continue: {lesson_title} ({path_title})",
                f"Kontynuuj: {lesson_title} ({path_title})",
            )

        return self.text(
            f"Start: {lesson_title} ({path_title})",
            f"Zacznij: {lesson_title} ({path_title})",
        )
