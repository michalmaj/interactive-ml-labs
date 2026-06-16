"""Tests for the native Feature Scaling Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.feature_scaling_scene import (
    MODELS,
    PRESETS,
    FeatureScalingLabScene,
    create_feature_scaling_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_feature_scaling_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Feature Scaling scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_scaling_lab_scene(AppContext())

        assert isinstance(scene, FeatureScalingLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_feature_scaling_scene_toggles_scaling_model_and_resets(monkeypatch) -> None:
    """S, M, preset keys, and R should update the scaling preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_scaling_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.model_name == MODELS[0]
        assert scene.scaling_enabled is False
        assert scene._accuracy_label() == "63%"
        assert scene._iterations_label() == "42"
        assert scene._range_ratio_label() == "2000:1"
        assert scene._diagnosis_key() == "dominance"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))

        assert command.kind == SceneCommandKind.NONE
        assert scene.scaling_enabled is True
        assert scene._accuracy_label() == "82%"
        assert scene._iterations_label() == "18"
        assert scene._range_ratio_label() == "1:1"
        assert scene._diagnosis_key() == "comparable"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.model_name == "Logistic Regression"
        assert scene.preset is PRESETS[2]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.model_name == MODELS[0]
        assert scene.scaling_enabled is False
    finally:
        pygame.quit()


def test_feature_scaling_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize operational labels while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_feature_scaling_lab_scene(context)

        assert scene._scaling_state_label() == "raw ranges"
        assert scene._diagnosis_label() == "dominacja cechy"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))

        assert scene._scaling_state_label() == "scaled"
        assert scene._diagnosis_label() == "porównywalne zakresy"
        assert "Zakresy są porównywalne" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_feature_scaling_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The scaling preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_scaling_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
