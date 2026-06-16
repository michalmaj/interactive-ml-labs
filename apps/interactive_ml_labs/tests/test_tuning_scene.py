"""Tests for the native Hyperparameter Tuning Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.tuning_scene import (
    DEFAULT_PARAM_INDEX,
    PRESETS,
    HyperparameterTuningLabScene,
    create_hyperparameter_tuning_lab_scene,
)


def test_tuning_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Tuning scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_hyperparameter_tuning_lab_scene(AppContext())

        assert isinstance(scene, HyperparameterTuningLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_tuning_scene_updates_parameter_and_resets(monkeypatch) -> None:
    """Parameter keys, preset keys, and R should update the tuning preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_hyperparameter_tuning_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.param_index == DEFAULT_PARAM_INDEX
        assert scene._parameter_label() == "k=7 (3/5)"
        assert scene._best_value_label() == "k=7"
        assert scene._diagnosis_key() == "candidate"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert command.kind == SceneCommandKind.NONE
        assert scene._parameter_label() == "k=15 (4/5)"
        assert scene._diagnosis_key() == "review"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))

        assert scene._parameter_label() == "k=31 (5/5)"
        assert scene._diagnosis_key() == "underfit"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.preset is PRESETS[1]
        assert scene.param_index == DEFAULT_PARAM_INDEX
        assert scene._best_value_label() == "max depth=4"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.param_index == DEFAULT_PARAM_INDEX
    finally:
        pygame.quit()


def test_tuning_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize diagnosis labels while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_hyperparameter_tuning_lab_scene(context)

        assert scene._diagnosis_label() == "kandydat z validation"
        assert "Validation ma tu maksimum" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

        assert scene._diagnosis_label() == "overfit"
    finally:
        pygame.quit()


def test_tuning_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The tuning preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_hyperparameter_tuning_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
