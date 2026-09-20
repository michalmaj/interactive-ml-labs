"""Catalog facade for shell access to registered learning content."""

from __future__ import annotations

from dataclasses import dataclass

from interactive_ml_labs.manifest import (
    CourseMapStep,
    DemoManifest,
    LearningPathManifest,
    LessonManifest,
    LocalizedText,
)
from interactive_ml_labs.registry import (
    COURSE_MAP_STEPS,
    DEMO_BY_ID,
    LEARNING_PATH_MANIFESTS,
    LESSON_BY_ID,
    LEVEL_NAMES,
    demos_for_level,
    levels_from_manifests,
)


@dataclass(frozen=True, slots=True)
class ShellCatalog:
    """Read-only catalog used by the unified app shell."""

    course_map_steps: tuple[CourseMapStep, ...] = COURSE_MAP_STEPS
    demos_by_id: dict[str, DemoManifest] | None = None
    learning_paths: tuple[LearningPathManifest, ...] = LEARNING_PATH_MANIFESTS
    lessons_by_id: dict[str, LessonManifest] | None = None
    level_names: dict[int, LocalizedText] | None = None

    def __post_init__(self) -> None:
        """Fill default mappings without sharing mutable defaults."""
        if self.demos_by_id is None:
            object.__setattr__(self, "demos_by_id", DEMO_BY_ID)
        if self.lessons_by_id is None:
            object.__setattr__(self, "lessons_by_id", LESSON_BY_ID)
        if self.level_names is None:
            object.__setattr__(self, "level_names", LEVEL_NAMES)

    def levels(self) -> tuple[int, ...]:
        """Return level numbers available in the demo registry."""
        return levels_from_manifests()

    def demos_for_level(self, level: int) -> tuple[DemoManifest, ...]:
        """Return demos belonging to one level."""
        return demos_for_level(level)

    def demo(self, demo_id: str) -> DemoManifest:
        """Return one demo manifest by id."""
        assert self.demos_by_id is not None
        return self.demos_by_id[demo_id]

    def lesson(self, lesson_id: str) -> LessonManifest:
        """Return one lesson manifest by id."""
        assert self.lessons_by_id is not None
        return self.lessons_by_id[lesson_id]

    def lessons_for_path(self, path: LearningPathManifest) -> tuple[LessonManifest, ...]:
        """Return lesson manifests for one learning path."""
        return tuple(self.lesson(lesson_id) for lesson_id in path.lesson_ids)

    def learning_path(self, index: int) -> LearningPathManifest:
        """Return one learning path by index."""
        return self.learning_paths[index]

    def course_map_step(self, index: int) -> CourseMapStep:
        """Return one course map step by index."""
        return self.course_map_steps[index]

    def course_map_path_for_step(self, step_index: int) -> LearningPathManifest:
        """Return the learning path referenced by one course-map step."""
        path_id = self.course_map_step(step_index).path_id
        return next(path for path in self.learning_paths if path.id == path_id)

    def level_name(self, level: int) -> LocalizedText:
        """Return localized level name metadata."""
        assert self.level_names is not None
        return self.level_names[level]
