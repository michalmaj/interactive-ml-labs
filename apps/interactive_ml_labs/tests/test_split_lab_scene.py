"""Tests for the native Train / Validation / Test Split Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.split_lab_scene import (
    CHOOSE_VALIDATION_TASK_ID,
    COMPARE_COMPLEXITY_TASK_ID,
    DEFAULT_COMPLEXITY_INDEX,
    PRESETS,
    SPLIT_LESSON_ID,
    TrainValidationTestLabScene,
    create_train_validation_test_lab_scene,
)


def test_split_lab_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Split lab scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_train_validation_test_lab_scene(AppContext())

        assert isinstance(scene, TrainValidationTestLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_split_lab_scene_updates_complexity_and_resets(monkeypatch) -> None:
    """Complexity keys, preset keys, and R should update the split preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_train_validation_test_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.complexity_index == DEFAULT_COMPLEXITY_INDEX
        assert scene._complexity_label() == "balanced (2/3)"
        assert scene._score_label(scene.metrics.validation) == "84%"
        assert scene._gap_label() == "+4%"
        assert scene._diagnosis_key() == "candidate"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert command.kind == SceneCommandKind.NONE
        assert scene._complexity_label() == "too flexible (3/3)"
        assert scene._diagnosis_key() == "overfit"
        assert scene._gap_label() == "+19%"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset is PRESETS[2]
        assert scene.preset.split_counts == (120, 40, 40)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.complexity_index == DEFAULT_COMPLEXITY_INDEX
    finally:
        pygame.quit()


def test_split_lab_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize operational labels while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_train_validation_test_lab_scene(context)

        assert scene._complexity_label() == "zbalansowany (2/3)"
        assert scene._diagnosis_label() == "kandydat z validation"
        assert "Validation jest tu najmocniejsze" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert scene._complexity_label() == "zbyt elastyczny (3/3)"
        assert scene._diagnosis_label() == "overfit"
    finally:
        pygame.quit()


def test_split_lab_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The split preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_train_validation_test_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_split_lab_scene_records_complexity_comparison_for_guided_lesson(monkeypatch) -> None:
    """Comparing complexity settings should complete the first guided task."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = SPLIT_LESSON_ID
        scene = create_train_validation_test_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        progress = context.progress.lessons[SPLIT_LESSON_ID]
        assert COMPARE_COMPLEXITY_TASK_ID in progress.completed_task_ids
        assert CHOOSE_VALIDATION_TASK_ID not in progress.completed_task_ids
        assert progress.completed is False
    finally:
        pygame.quit()


def test_split_lab_scene_completes_guided_lesson_at_validation_candidate(monkeypatch) -> None:
    """Returning to the best validation candidate should complete the lesson."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.selected_lesson_id = SPLIT_LESSON_ID
        scene = create_train_validation_test_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0))

        progress = context.progress.lessons[SPLIT_LESSON_ID]
        assert progress.completed_task_ids >= {
            COMPARE_COMPLEXITY_TASK_ID,
            CHOOSE_VALIDATION_TASK_ID,
        }
        assert progress.completed is True
    finally:
        pygame.quit()


def test_split_lab_scene_ignores_progress_outside_guided_lesson(monkeypatch) -> None:
    """Standalone Split Lab use should not mutate guided lesson progress."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_train_validation_test_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0))

        assert SPLIT_LESSON_ID not in context.progress.lessons
    finally:
        pygame.quit()
