"""Persistent lesson progress for the unified learning platform."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROGRESS_FILE_NAME = "progress.json"
PROGRESS_SCHEMA_VERSION = 1


@dataclass(slots=True)
class LessonProgress:
    """Progress tracked for one guided lesson."""

    started: bool = False
    theory_visited: bool = False
    completed_task_ids: set[str] = field(default_factory=set)
    completed: bool = False


@dataclass(slots=True)
class AppProgress:
    """Persistent progress across guided learning paths."""

    lessons: dict[str, LessonProgress] = field(default_factory=dict)
    revision: int = 0

    def lesson(self, lesson_id: str) -> LessonProgress:
        """Return existing progress for a lesson, creating it when needed."""
        if lesson_id not in self.lessons:
            self.lessons[lesson_id] = LessonProgress()
            self.revision += 1

        return self.lessons[lesson_id]

    def mark_started(self, lesson_id: str) -> None:
        """Mark a lesson as started."""
        progress = self.lesson(lesson_id)
        if progress.started:
            return

        progress.started = True
        self.revision += 1

    def mark_theory_visited(self, lesson_id: str) -> None:
        """Mark the theory screen as visited for a lesson."""
        progress = self.lesson(lesson_id)
        changed = not progress.started or not progress.theory_visited
        if not changed:
            return

        progress.started = True
        progress.theory_visited = True
        self.revision += 1

    def complete_task(self, lesson_id: str, task_id: str) -> None:
        """Mark one lesson task as completed."""
        progress = self.lesson(lesson_id)
        changed = not progress.started or task_id not in progress.completed_task_ids
        if not changed:
            return

        progress.started = True
        progress.completed_task_ids.add(task_id)
        self.revision += 1

    def mark_completed(self, lesson_id: str) -> None:
        """Mark a lesson as completed."""
        progress = self.lesson(lesson_id)
        changed = not progress.started or not progress.completed
        if not changed:
            return

        progress.started = True
        progress.completed = True
        self.revision += 1


def default_progress_path() -> Path:
    """Return the per-user progress path for the shell."""
    if os.name == "nt":
        config_root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
        config_root = Path(xdg_config_home)
    elif sys.platform == "darwin":
        config_root = Path.home() / "Library" / "Application Support"
    else:
        config_root = Path.home() / ".config"

    return config_root / "interactive-ml-labs" / PROGRESS_FILE_NAME


def progress_to_json(progress: AppProgress) -> dict[str, object]:
    """Serialize progress to a JSON-friendly mapping."""
    return {
        "version": PROGRESS_SCHEMA_VERSION,
        "lessons": {
            lesson_id: {
                "started": lesson_progress.started,
                "theory_visited": lesson_progress.theory_visited,
                "completed_task_ids": sorted(lesson_progress.completed_task_ids),
                "completed": lesson_progress.completed,
            }
            for lesson_id, lesson_progress in sorted(progress.lessons.items())
        },
    }


def progress_from_json(data: object) -> AppProgress:
    """Build progress from JSON data, ignoring malformed records."""
    progress = AppProgress()
    if not isinstance(data, dict):
        return progress

    lessons = data.get("lessons")
    if not isinstance(lessons, dict):
        return progress

    for lesson_id, lesson_data in lessons.items():
        if not isinstance(lesson_id, str) or not isinstance(lesson_data, dict):
            continue

        completed_task_ids = lesson_data.get("completed_task_ids", [])
        if not isinstance(completed_task_ids, list):
            completed_task_ids = []

        progress.lessons[lesson_id] = LessonProgress(
            started=lesson_data.get("started") is True,
            theory_visited=lesson_data.get("theory_visited") is True,
            completed_task_ids={
                task_id for task_id in completed_task_ids if isinstance(task_id, str)
            },
            completed=lesson_data.get("completed") is True,
        )

    return progress


def load_app_progress(path: Path | None = None) -> AppProgress:
    """Load persisted progress, returning empty progress when absent or invalid."""
    progress_path = path or default_progress_path()
    try:
        with progress_path.open(encoding="utf-8") as file:
            return progress_from_json(json.load(file))
    except (OSError, json.JSONDecodeError):
        return AppProgress()


def save_app_progress(progress: AppProgress, path: Path | None = None) -> None:
    """Persist progress best-effort without interrupting the app on write errors."""
    progress_path = path or default_progress_path()
    try:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("w", encoding="utf-8") as file:
            json.dump(progress_to_json(progress), file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError:
        return
