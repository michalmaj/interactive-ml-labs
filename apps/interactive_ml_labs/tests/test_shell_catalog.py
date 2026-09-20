"""Tests for the unified shell catalog facade."""

from interactive_ml_labs.registry import (
    COURSE_MAP_STEPS,
    DEMO_BY_ID,
    LEARNING_PATH_MANIFESTS,
    LESSON_BY_ID,
    LEVEL_NAMES,
    demos_for_level,
    levels_from_manifests,
)
from interactive_ml_labs.shell_catalog import ShellCatalog


def test_shell_catalog_exposes_registry_level_metadata() -> None:
    """Catalog should preserve current level and demo ordering."""
    catalog = ShellCatalog()

    assert catalog.levels() == levels_from_manifests()
    assert catalog.demos_for_level(1) == demos_for_level(1)
    assert catalog.level_name(1) == LEVEL_NAMES[1]


def test_shell_catalog_exposes_demo_and_lesson_by_id() -> None:
    """Catalog should expose demo and lesson lookups through one facade."""
    catalog = ShellCatalog()

    assert catalog.demo("gradient_descent_playground") == DEMO_BY_ID["gradient_descent_playground"]
    assert catalog.lesson("error_gradient_descent") == LESSON_BY_ID["error_gradient_descent"]


def test_shell_catalog_exposes_learning_path_lessons() -> None:
    """Catalog should resolve learning-path lesson ids to lesson manifests."""
    catalog = ShellCatalog()
    path = LEARNING_PATH_MANIFESTS[0]

    assert catalog.all_learning_paths() == LEARNING_PATH_MANIFESTS
    assert catalog.learning_path_count() == len(LEARNING_PATH_MANIFESTS)
    assert catalog.learning_path(0) == path
    assert catalog.learning_path_index(path) == 0
    assert catalog.lessons_for_path(path) == tuple(
        LESSON_BY_ID[lesson_id] for lesson_id in path.lesson_ids
    )
    assert catalog.lesson_exists(path.lesson_ids[0])
    assert not catalog.lesson_exists("missing_lesson")


def test_shell_catalog_exposes_next_learning_path() -> None:
    """Catalog should expose adjacent guided paths without leaking registry globals."""
    catalog = ShellCatalog()

    assert catalog.next_learning_path(LEARNING_PATH_MANIFESTS[0]) == LEARNING_PATH_MANIFESTS[1]
    assert catalog.next_learning_path(LEARNING_PATH_MANIFESTS[-1]) is None


def test_shell_catalog_exposes_course_map_paths() -> None:
    """Catalog should resolve course-map steps to their learning paths."""
    catalog = ShellCatalog()
    step = COURSE_MAP_STEPS[0]

    assert catalog.course_map_step_count() == len(COURSE_MAP_STEPS)
    assert catalog.course_map_menu_item_count() == len(COURSE_MAP_STEPS) + 1
    assert catalog.course_map_step(0) == step
    assert catalog.course_map_path_for_step(0).id == step.path_id
