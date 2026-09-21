"""Tests for home progress view-data builders."""

from interactive_ml_labs.progress import AppProgress
from interactive_ml_labs.registry import LEARNING_PATH_MANIFESTS, LESSON_BY_ID
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_home_progress_view_data import ShellHomeProgressViewData
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary


def _text_en(en: str, pl: str) -> str:
    """Return English text in tests."""
    _ = pl
    return en


def _text_pl(en: str, pl: str) -> str:
    """Return Polish text in tests."""
    _ = en
    return pl


def _view_data(progress: AppProgress, *, language: str = "en") -> ShellHomeProgressViewData:
    """Return a test home progress view-data builder."""
    catalog = ShellCatalog()
    return ShellHomeProgressViewData(
        catalog=catalog,
        progress=ShellProgressSummary(catalog=catalog, progress=progress),
        language=language,
        text=_text_pl if language == "pl" else _text_en,
    )


def _guided_lesson_total() -> int:
    """Return total guided lesson count across registered paths."""
    return sum(len(path.lesson_ids) for path in LEARNING_PATH_MANIFESTS)


def _guided_task_total() -> int:
    """Return total guided task count."""
    return sum(
        len(LESSON_BY_ID[lesson_id].tasks)
        for path in LEARNING_PATH_MANIFESTS
        for lesson_id in path.lesson_ids
    )


def test_home_progress_view_data_summarizes_empty_progress() -> None:
    """Home progress view-data should summarize all guided paths."""
    first_path = LEARNING_PATH_MANIFESTS[0]
    first_lesson = LESSON_BY_ID[first_path.lesson_ids[0]]
    details = _view_data(AppProgress()).home_progress_details()

    assert details.lines == [
        f"Lessons: 0/{_guided_lesson_total()} completed",
        f"Tasks: 0/{_guided_task_total()} completed",
        f"Badges: 0/{_guided_lesson_total()} unlocked",
        f"Start: {first_lesson.title.en} ({first_path.title.en})",
    ]


def test_home_progress_view_data_tracks_completed_work() -> None:
    """Home progress view-data should expose labels and numeric progress."""
    progress = AppProgress()
    path = LEARNING_PATH_MANIFESTS[0]
    lesson = LESSON_BY_ID[path.lesson_ids[0]]
    progress.complete_task(lesson.id, lesson.tasks[0].id)
    progress.mark_completed(lesson.id)

    metrics = _view_data(progress).home_progress_metrics()

    assert [(metric.label, metric.completed_count, metric.total_count) for metric in metrics] == [
        (f"Lessons: 1/{_guided_lesson_total()} completed", 1, _guided_lesson_total()),
        (f"Tasks: 1/{_guided_task_total()} completed", 1, _guided_task_total()),
        (f"Badges: 1/{_guided_lesson_total()} unlocked", 1, _guided_lesson_total()),
    ]


def test_home_progress_view_data_localizes_polish_next_action() -> None:
    """Home progress view-data should use natural Polish labels."""
    first_path = LEARNING_PATH_MANIFESTS[0]
    first_lesson = LESSON_BY_ID[first_path.lesson_ids[0]]
    details = _view_data(AppProgress(), language="pl").home_progress_details()

    assert details.lines == [
        f"Lekcje: 0/{_guided_lesson_total()} ukończone",
        f"Zadania: 0/{_guided_task_total()} ukończone",
        f"Odznaki: 0/{_guided_lesson_total()} zdobyte",
        f"Zacznij: {first_lesson.title.pl} ({first_path.title.pl})",
    ]
