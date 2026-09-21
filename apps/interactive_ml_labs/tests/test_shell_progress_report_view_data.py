"""Tests for progress report view-data builders."""

from interactive_ml_labs.progress import AppProgress
from interactive_ml_labs.registry import LEARNING_PATH_MANIFESTS, LESSON_BY_ID
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_learning_path_view_data import ShellLearningPathViewData
from interactive_ml_labs.shell_progress_report_view_data import ShellProgressReportViewData
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary


def _text_en(en: str, pl: str) -> str:
    """Return English text in tests."""
    _ = pl
    return en


def _text_pl(en: str, pl: str) -> str:
    """Return Polish text in tests."""
    _ = en
    return pl


def _view_data(progress: AppProgress, *, language: str = "en") -> ShellProgressReportViewData:
    """Return a test progress report view-data builder."""
    catalog = ShellCatalog()
    summary = ShellProgressSummary(catalog=catalog, progress=progress)
    text = _text_pl if language == "pl" else _text_en
    learning_paths = ShellLearningPathViewData(
        catalog=catalog,
        progress=summary,
        language=language,
        text=text,
    )
    return ShellProgressReportViewData(
        catalog=catalog,
        progress=summary,
        learning_paths=learning_paths,
        language=language,
        text=text,
    )


def test_progress_report_summarizes_completed_work() -> None:
    """Progress report should count paths, lessons, tasks, and badges."""
    progress = AppProgress()
    path = LEARNING_PATH_MANIFESTS[0]
    first_lesson_id = path.lesson_ids[0]
    first_task_id = LESSON_BY_ID[first_lesson_id].tasks[0].id
    progress.mark_theory_visited(first_lesson_id)
    progress.complete_task(first_lesson_id, first_task_id)
    progress.mark_completed(first_lesson_id)
    view_data = _view_data(progress)
    total_lessons = sum(len(path.lesson_ids) for path in LEARNING_PATH_MANIFESTS)
    total_tasks = sum(
        len(LESSON_BY_ID[lesson_id].tasks)
        for path in LEARNING_PATH_MANIFESTS
        for lesson_id in path.lesson_ids
    )
    total_badges = sum(
        1
        for path in LEARNING_PATH_MANIFESTS
        for lesson_id in path.lesson_ids
        if LESSON_BY_ID[lesson_id].completion_badge is not None
    )

    details = view_data.progress_report_details()

    assert details.title == "Progress report"
    assert [metric.label for metric in details.metrics] == [
        f"Paths completed: 0/{len(LEARNING_PATH_MANIFESTS)}",
        f"Lessons completed: 1/{total_lessons}",
        f"Tasks completed: 1/{total_tasks}",
        f"Badges unlocked: 1/{total_badges}",
    ]
    assert details.paths[0].status_label == "In progress"
    assert details.paths[0].lesson_label == "Lessons: 1/4"
    assert details.paths[0].task_label == "Tasks: 1/8"
    assert details.paths[0].badge_label == "Badges: 1/4"
    assert any(badge.unlocked for badge in details.badges)
    assert details.reflection_lines == [
        "No completed lesson has a reflection mark yet.",
    ]
    assert details.review_lines == [
        "No lessons are marked for review.",
    ]


def test_progress_report_lists_completed_learning_claims() -> None:
    """Completed lessons should unlock concise explanation lines."""
    progress = AppProgress()
    lesson_id = LEARNING_PATH_MANIFESTS[0].lesson_ids[0]
    progress.mark_completed(lesson_id)
    view_data = _view_data(progress, language="pl")

    details = view_data.progress_report_details()

    assert details.explain_heading == "Co już umiesz wyjaśnić"
    assert len(details.explain_lines) == 1
    assert details.explain_lines[0].startswith("Po lekcji")
    assert "residuals" in details.explain_lines[0]


def test_progress_report_has_empty_state_explanation() -> None:
    """Empty progress should still explain how to unlock the first summary."""
    details = _view_data(AppProgress(), language="pl").progress_report_details()

    assert details.explain_lines == [
        "Ukończ pierwszą prowadzoną lekcję, a pojawi się tu krótkie podsumowanie.",
    ]


def test_progress_report_summarizes_lesson_reflection_statuses() -> None:
    """Progress report should summarize saved self-check reflection statuses."""
    progress = AppProgress()
    first_lesson_id = LEARNING_PATH_MANIFESTS[0].lesson_ids[0]
    second_lesson_id = LEARNING_PATH_MANIFESTS[0].lesson_ids[1]
    progress.set_reflection_status(first_lesson_id, "understood")
    progress.set_reflection_status(second_lesson_id, "review")

    details = _view_data(progress, language="pl").progress_report_details()

    assert details.reflection_heading == "Samoocena"
    assert details.reflection_lines == [
        "Rozumiem: 1",
        "Do powtórki: 1",
    ]
    assert details.review_heading == "Do powtórki"
    assert details.review_lines == [
        (f"{LESSON_BY_ID[second_lesson_id].title.pl} ({LEARNING_PATH_MANIFESTS[0].title.pl})"),
    ]
