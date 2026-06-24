"""Tests for the native Gaussian Mixture Intro Lab scene."""

import pygame
import pytest
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.gaussian_mixture_scene import (
    ADJUST_COMPONENT_COUNT_TASK_ID,
    COMPARE_SOFT_ASSIGNMENTS_TASK_ID,
    GAUSSIAN_MIXTURE_LESSON_ID,
    MAX_COMPONENTS,
    MIN_COMPONENTS,
    PRESETS,
    GaussianMixtureIntroLabScene,
    create_gaussian_mixture_intro_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_gaussian_mixture_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Gaussian Mixture Intro should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gaussian_mixture_intro_lab_scene(AppContext())

        assert isinstance(scene, GaussianMixtureIntroLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_responsibilities_are_soft_and_normalized(monkeypatch) -> None:
    """Query responsibilities should form a normalized soft assignment."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gaussian_mixture_intro_lab_scene(AppContext())

        responsibilities = scene.responsibilities_for_query()

        assert len(responsibilities) == scene.component_count
        assert sum(responsibilities) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in responsibilities)
        assert max(responsibilities) < 0.85
        assert scene._diagnosis_key() in {"ambiguous", "mixed"}
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_controls_components_query_and_reset(monkeypatch) -> None:
    """Preset, component, query, and reset controls should update the preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gaussian_mixture_intro_lab_scene(AppContext())
        start_query = scene.query

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert command.kind == SceneCommandKind.NONE
        assert scene.query.x > start_query.x

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.component_count == 3

        for _ in range(10):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.component_count == MAX_COMPONENTS

        for _ in range(10):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert scene.component_count == MIN_COMPONENTS

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        assert scene.preset is PRESETS[2]
        assert scene.query == PRESETS[2].query_start

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))
        assert scene.show_density is False
        assert scene.hard_assignment is True
        assert scene._mode_label() == "hard assignment"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert scene.preset is PRESETS[0]
        assert scene.component_count == 2
        assert scene.show_density is True
        assert scene.hard_assignment is False
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_records_soft_assignment_task(monkeypatch) -> None:
    """Moving the query point should complete the soft-assignment task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=GAUSSIAN_MIXTURE_LESSON_ID)
        scene = create_gaussian_mixture_intro_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        lesson_progress = context.progress.lessons[GAUSSIAN_MIXTURE_LESSON_ID]

        assert COMPARE_SOFT_ASSIGNMENTS_TASK_ID in lesson_progress.completed_task_ids
        assert ADJUST_COMPONENT_COUNT_TASK_ID not in lesson_progress.completed_task_ids
        assert lesson_progress.completed is False
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_records_component_count_task(monkeypatch) -> None:
    """Changing component count should complete the component-count task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=GAUSSIAN_MIXTURE_LESSON_ID)
        scene = create_gaussian_mixture_intro_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        lesson_progress = context.progress.lessons[GAUSSIAN_MIXTURE_LESSON_ID]

        assert ADJUST_COMPONENT_COUNT_TASK_ID in lesson_progress.completed_task_ids
        assert COMPARE_SOFT_ASSIGNMENTS_TASK_ID not in lesson_progress.completed_task_ids
        assert lesson_progress.completed is False
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_completes_guided_lesson(monkeypatch) -> None:
    """Completing both guided tasks should unlock the lesson completion state."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=GAUSSIAN_MIXTURE_LESSON_ID)
        scene = create_gaussian_mixture_intro_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        lesson_progress = context.progress.lessons[GAUSSIAN_MIXTURE_LESSON_ID]

        assert {
            COMPARE_SOFT_ASSIGNMENTS_TASK_ID,
            ADJUST_COMPONENT_COUNT_TASK_ID,
        }.issubset(lesson_progress.completed_task_ids)
        assert lesson_progress.completed is True
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_ignores_progress_outside_guided_lesson(monkeypatch) -> None:
    """Standalone demo use should not write guided lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_gaussian_mixture_intro_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert GAUSSIAN_MIXTURE_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize labels while keeping core GMM terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_gaussian_mixture_intro_lab_scene(context)

        assert scene._mode_label() == "soft responsibilities"
        assert "punkt" in scene._diagnosis_label() or "sygnał" in scene._diagnosis_label()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h))

        assert scene._mode_label() == "hard assignment"
        assert "ukrywa uncertainty" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_gaussian_mixture_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The Gaussian Mixture preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_gaussian_mixture_intro_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
