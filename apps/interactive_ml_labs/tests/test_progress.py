"""Tests for guided lesson progress persistence."""

import json

from interactive_ml_labs.progress import (
    AppProgress,
    LessonProgress,
    load_app_progress,
    progress_from_json,
    progress_to_json,
    save_app_progress,
)


def test_app_progress_creates_lesson_records_on_demand() -> None:
    """Progress helpers should create lesson records lazily."""
    progress = AppProgress()

    progress.mark_started("lesson_one")
    progress.mark_theory_visited("lesson_one")
    progress.complete_task("lesson_one", "task_a")
    progress.mark_completed("lesson_one")

    lesson = progress.lessons["lesson_one"]
    assert lesson.started is True
    assert lesson.theory_visited is True
    assert lesson.completed_task_ids == {"task_a"}
    assert lesson.completed is True


def test_progress_serialization_round_trips_lesson_state() -> None:
    """Lesson progress should round-trip through JSON-friendly data."""
    progress = AppProgress(
        lessons={
            "lesson_one": LessonProgress(
                started=True,
                theory_visited=True,
                completed_task_ids={"task_b", "task_a"},
                completed=False,
            ),
        },
    )

    data = progress_to_json(progress)
    loaded = progress_from_json(data)

    assert data == {
        "version": 1,
        "lessons": {
            "lesson_one": {
                "started": True,
                "theory_visited": True,
                "completed_task_ids": ["task_a", "task_b"],
                "completed": False,
            },
        },
    }
    assert loaded.lessons["lesson_one"].completed_task_ids == {"task_a", "task_b"}


def test_progress_from_json_ignores_malformed_records() -> None:
    """Malformed progress records should not break app startup."""
    progress = progress_from_json(
        {
            "lessons": {
                "valid": {
                    "started": True,
                    "theory_visited": "yes",
                    "completed_task_ids": ["a", 2, "b"],
                    "completed": False,
                },
                42: {"started": True},
                "bad": "not a mapping",
            },
        },
    )

    assert set(progress.lessons) == {"valid"}
    assert progress.lessons["valid"].started is True
    assert progress.lessons["valid"].theory_visited is False
    assert progress.lessons["valid"].completed_task_ids == {"a", "b"}


def test_load_app_progress_returns_empty_when_file_is_missing(tmp_path) -> None:
    """Missing progress files should start with empty progress."""
    assert load_app_progress(tmp_path / "missing" / "progress.json") == AppProgress()


def test_save_and_load_app_progress_round_trip(tmp_path) -> None:
    """Progress should persist to a small JSON file."""
    progress_path = tmp_path / "interactive-ml-labs" / "progress.json"
    progress = AppProgress()
    progress.mark_theory_visited("lesson_one")
    progress.complete_task("lesson_one", "task_a")

    save_app_progress(progress, progress_path)
    loaded = load_app_progress(progress_path)

    assert loaded.lessons["lesson_one"].started is True
    assert loaded.lessons["lesson_one"].theory_visited is True
    assert loaded.lessons["lesson_one"].completed_task_ids == {"task_a"}
    assert json.loads(progress_path.read_text(encoding="utf-8")) == progress_to_json(progress)


def test_load_app_progress_returns_empty_for_invalid_json(tmp_path) -> None:
    """A corrupt progress file should be ignored."""
    progress_path = tmp_path / "progress.json"
    progress_path.write_text("{", encoding="utf-8")

    assert load_app_progress(progress_path) == AppProgress()
