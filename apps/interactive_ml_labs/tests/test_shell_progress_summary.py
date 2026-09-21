"""Tests for shell progress summary helpers."""

from interactive_ml_labs.progress import AppProgress
from interactive_ml_labs.registry import LEARNING_PATH_MANIFESTS
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary


def test_shell_progress_summary_reports_empty_path_counts() -> None:
    """Empty progress should produce zero completed counts and first lesson as next."""
    catalog = ShellCatalog()
    progress = AppProgress()
    summary = ShellProgressSummary(catalog=catalog, progress=progress)
    path = LEARNING_PATH_MANIFESTS[0]

    assert summary.completed_learning_path_lesson_count(path) == 0
    assert summary.completed_learning_path_task_count(path) == 0
    assert summary.visited_learning_path_theory_count(path) == 0
    assert summary.unlocked_learning_path_badge_count(path) == 0
    assert summary.started_learning_path_lesson_count(path) == 0
    assert summary.learning_path_task_count(path) == 8
    assert summary.learning_path_badge_count(path) == 4
    assert summary.next_learning_path_lesson(path) == catalog.lesson(path.lesson_ids[0])
    assert summary.next_learning_path_lesson_index(path) == 0


def test_shell_progress_summary_tracks_started_completed_and_tasks() -> None:
    """Summary should combine started, task, theory, lesson, and badge progress."""
    catalog = ShellCatalog()
    progress = AppProgress()
    summary = ShellProgressSummary(catalog=catalog, progress=progress)
    path = LEARNING_PATH_MANIFESTS[0]
    first_lesson = catalog.lesson(path.lesson_ids[0])

    progress.mark_started(first_lesson.id)
    progress.mark_theory_visited(first_lesson.id)
    progress.complete_task(first_lesson.id, first_lesson.tasks[0].id)
    progress.mark_completed(path.lesson_ids[1])

    assert summary.has_lesson_progress(first_lesson.id)
    assert summary.completed_lesson_task_ids(first_lesson) == {first_lesson.tasks[0].id}
    assert summary.completed_learning_path_lesson_count(path) == 1
    assert summary.completed_learning_path_task_count(path) == 1
    assert summary.visited_learning_path_theory_count(path) == 1
    assert summary.unlocked_learning_path_badge_count(path) == 1
    assert summary.started_learning_path_lesson_count(path) == 2
    assert summary.next_learning_path_lesson(path) == first_lesson
    assert summary.next_learning_path_lesson_index(path) == 0


def test_shell_progress_summary_finds_next_path_step() -> None:
    """Summary should find the first path that still has incomplete lessons."""
    catalog = ShellCatalog()
    progress = AppProgress()
    summary = ShellProgressSummary(catalog=catalog, progress=progress)
    first_path = LEARNING_PATH_MANIFESTS[0]

    for lesson_id in first_path.lesson_ids:
        progress.mark_completed(lesson_id)

    next_path, next_lesson = summary.next_learning_path_step()

    assert next_path == LEARNING_PATH_MANIFESTS[1]
    assert next_lesson == catalog.lesson(LEARNING_PATH_MANIFESTS[1].lesson_ids[0])
