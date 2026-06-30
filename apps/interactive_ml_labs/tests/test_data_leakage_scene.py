"""Tests for the native Data Leakage Lab scene."""

import pygame
from interactive_ml_labs.data_leakage_scene import (
    COMPARE_SCENARIOS_TASK_ID,
    LEAKAGE_LESSON_ID,
    PRESETS,
    REMOVE_LEAKAGE_TASK_ID,
    DataLeakageLabScene,
    create_data_leakage_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_data_leakage_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Data Leakage scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_data_leakage_lab_scene(AppContext())

        assert isinstance(scene, DataLeakageLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_data_leakage_scene_toggles_leakage_and_resets(monkeypatch) -> None:
    """L, preset keys, and R should update the leakage scenario."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_data_leakage_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.leakage_enabled is True
        assert scene._train_accuracy_label() == "99%"
        assert scene._test_accuracy_label() == "98%"
        assert scene._diagnosis_key() == "too_good"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))

        assert command.kind == SceneCommandKind.NONE
        assert scene.leakage_enabled is False
        assert scene._train_accuracy_label() == "74%"
        assert scene._test_accuracy_label() == "71%"
        assert scene._leakage_feature_label() == "removed"
        assert scene._diagnosis_key() == "realistic"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset is PRESETS[2]
        assert scene._test_accuracy_label() == "72%"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.leakage_enabled is True
    finally:
        pygame.quit()


def test_data_leakage_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should avoid English-only operational labels."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_data_leakage_lab_scene(context)

        assert scene._model_state_label() == "cecha leakage włączona"
        assert scene._diagnosis_label() == "zbyt dobre, by ufać"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))

        assert scene._model_state_label() == "cecha leakage usunięta"
        assert scene._leakage_feature_label() == "usunięta"
        assert scene._diagnosis_label() == "bardziej realistyczne"
    finally:
        pygame.quit()


def test_data_leakage_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The leakage preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_data_leakage_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_data_leakage_scene_records_removal_for_guided_lesson(monkeypatch) -> None:
    """Removing leakage should complete the first guided lesson task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = LEAKAGE_LESSON_ID
        scene = create_data_leakage_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))

        progress = context.progress.lessons[LEAKAGE_LESSON_ID]
        assert REMOVE_LEAKAGE_TASK_ID in progress.completed_task_ids
        assert COMPARE_SCENARIOS_TASK_ID not in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_data_leakage_scene_completes_guided_lesson_after_scenario_comparison(
    monkeypatch,
) -> None:
    """Removing leakage and visiting another scenario should complete the lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = LEAKAGE_LESSON_ID
        scene = create_data_leakage_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        progress = context.progress.lessons[LEAKAGE_LESSON_ID]
        assert progress.completed_task_ids >= {
            REMOVE_LEAKAGE_TASK_ID,
            COMPARE_SCENARIOS_TASK_ID,
        }
        assert progress.completed is True
    finally:
        pygame.quit()


def test_data_leakage_scene_ignores_progress_outside_guided_lesson(monkeypatch) -> None:
    """Standalone Data Leakage Lab use should not mutate guided lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_data_leakage_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert LEAKAGE_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
