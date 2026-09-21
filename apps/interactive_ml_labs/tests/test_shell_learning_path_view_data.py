"""Tests for guided learning path view-data builders."""

from interactive_ml_labs.progress import AppProgress
from interactive_ml_labs.registry import LEARNING_PATH_MANIFESTS
from interactive_ml_labs.shell_catalog import ShellCatalog
from interactive_ml_labs.shell_learning_path_view_data import ShellLearningPathViewData
from interactive_ml_labs.shell_progress_summary import ShellProgressSummary


def _text_en(en: str, pl: str) -> str:
    """Return English text in tests."""
    _ = pl
    return en


def _text_pl(en: str, pl: str) -> str:
    """Return Polish text in tests."""
    _ = en
    return pl


def _view_data(
    progress: AppProgress,
    *,
    language: str = "en",
) -> ShellLearningPathViewData:
    """Return a test view-data builder."""
    catalog = ShellCatalog()
    return ShellLearningPathViewData(
        catalog=catalog,
        progress=ShellProgressSummary(catalog=catalog, progress=progress),
        language=language,
        text=_text_pl if language == "pl" else _text_en,
    )


def test_learning_path_view_data_builds_path_details() -> None:
    """View-data builder should produce renderer-ready learning path details."""
    progress = AppProgress()
    path = LEARNING_PATH_MANIFESTS[0]
    progress.mark_completed(path.lesson_ids[0])
    view_data = _view_data(progress)

    details = view_data.learning_path_details(path)

    assert details.lesson_count_label == "4 lessons"
    assert details.course_map_heading == "Course map"
    assert [metric.label for metric in details.progress_metrics] == [
        "Lessons: 1/4 completed",
        "Tasks: 0/8 completed",
        "Theory: 0/4 visited",
        "Badges: 1/4 unlocked",
    ]
    assert [
        (metric.completed_count, metric.total_count) for metric in details.progress_metrics
    ] == [(1, 4), (0, 8), (0, 4), (1, 4)]
    assert details.status_label == "In progress"
    assert details.next_action_label == "Next action: start Let an algorithm reduce loss"
    assert "[x] Residual Reader" in details.badge_labels
    assert "[ ] Loss Navigator" in details.badge_labels


def test_learning_path_view_data_builds_badge_gallery_paths() -> None:
    """View-data builder should produce badge gallery groups with unlock state."""
    progress = AppProgress()
    path = LEARNING_PATH_MANIFESTS[0]
    progress.mark_completed(path.lesson_ids[0])
    view_data = _view_data(progress)

    gallery_paths = view_data.badge_gallery_paths()

    assert gallery_paths[0].title == path.title.en
    assert gallery_paths[0].progress_label == "Badges: 1/4 unlocked"
    assert [badge.label for badge in gallery_paths[0].badges[:2]] == [
        "Residual Reader",
        "Loss Navigator",
    ]
    assert [badge.unlocked for badge in gallery_paths[0].badges[:2]] == [True, False]


def test_learning_path_view_data_localizes_polish_badges_and_lessons() -> None:
    """View-data builder should preserve natural Polish learning path copy."""
    progress = AppProgress()
    path = next(path for path in LEARNING_PATH_MANIFESTS if path.id == "trustworthy_models")
    view_data = _view_data(progress, language="pl")

    details = view_data.learning_path_details(path)

    assert "[ ] Strażnik test set" in details.badge_labels
    assert "[ ] Detektyw leakage" in details.badge_labels
    assert "[ ] Strażnik modelu" in details.badge_labels
    assert "naprawdę coś znaczy" not in " ".join(
        [details.summary, *details.badge_labels, *details.lesson_labels],
    )
