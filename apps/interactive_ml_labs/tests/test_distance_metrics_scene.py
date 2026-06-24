"""Tests for the native Distance Metrics Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.distance_metrics_scene import (
    COMPARE_METRICS_TASK_ID,
    DISTANCE_METRICS_LESSON_ID,
    METRICS,
    MOVE_QUERY_TASK_ID,
    PRESETS,
    QUERY_STEP,
    DistanceMetric,
    DistanceMetricsLabScene,
    create_distance_metrics_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_distance_metrics_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Distance metrics scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_distance_metrics_lab_scene(AppContext())

        assert isinstance(scene, DistanceMetricsLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_distance_metrics_scene_moves_query_cycles_metric_and_resets(monkeypatch) -> None:
    """Arrow keys, M, preset keys, and R should update the distance preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_distance_metrics_lab_scene(AppContext())
        start_x = scene.query_x
        start_y = scene.query_y

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert command.kind == SceneCommandKind.NONE
        assert scene.query_x == start_x + QUERY_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        assert scene.query_y == start_y + QUERY_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert scene.metric is DistanceMetric.MANHATTAN
        assert scene._metric_label() == "Manhattan"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset is PRESETS[2]
        assert (scene.query_x, scene.query_y) == PRESETS[2].query
        assert scene.metric is METRICS[1]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.metric is DistanceMetric.EUCLIDEAN
        assert (scene.query_x, scene.query_y) == PRESETS[0].query
    finally:
        pygame.quit()


def test_distance_metrics_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize diagnosis text while keeping metric names."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_distance_metrics_lab_scene(context)

        assert scene._metric_label() == "Euclidean"
        assert scene._diagnosis_label() in {"ten sam nearest", "metryka zmieniła nearest"}

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert "Manhattan distance" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_distance_metrics_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The distance metrics preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_distance_metrics_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_distance_metrics_scene_records_query_task_for_guided_lesson(monkeypatch) -> None:
    """Moving the query point should complete the first guided lesson task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = DISTANCE_METRICS_LESSON_ID
        scene = create_distance_metrics_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        progress = context.progress.lessons[DISTANCE_METRICS_LESSON_ID]
        assert MOVE_QUERY_TASK_ID in progress.completed_task_ids
        assert COMPARE_METRICS_TASK_ID not in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_distance_metrics_scene_records_metric_task_and_completes_lesson(
    monkeypatch,
) -> None:
    """Comparing metrics after moving the query should complete the lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = DISTANCE_METRICS_LESSON_ID
        scene = create_distance_metrics_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        progress = context.progress.lessons[DISTANCE_METRICS_LESSON_ID]
        assert progress.completed_task_ids >= {MOVE_QUERY_TASK_ID, COMPARE_METRICS_TASK_ID}
        assert progress.completed is True
    finally:
        pygame.quit()


def test_distance_metrics_scene_ignores_task_progress_outside_guided_lesson(
    monkeypatch,
) -> None:
    """Standalone Distance Metrics interactions should not write lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_distance_metrics_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert DISTANCE_METRICS_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
