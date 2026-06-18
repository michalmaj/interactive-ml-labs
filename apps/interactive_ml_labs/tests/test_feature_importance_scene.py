"""Tests for the native Feature Importance Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.feature_importance_scene import (
    PRESETS,
    FeatureImportanceLabScene,
    ImportanceMethod,
    create_feature_importance_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_feature_importance_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Feature Importance scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_importance_lab_scene(AppContext())

        assert isinstance(scene, FeatureImportanceLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_feature_importance_scene_switches_method_and_resets(monkeypatch) -> None:
    """M, C, L, preset keys, and R should update the interpretation preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_importance_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.method == ImportanceMethod.PERMUTATION
        assert scene.top_feature().name == "income"
        assert scene._diagnosis_key() == "correlated"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert command.kind == SceneCommandKind.NONE
        assert scene.method == ImportanceMethod.MODEL
        assert scene._method_label() == "model importance"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.show_correlation_groups is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        assert scene.show_leakage_warning is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        assert scene.preset is PRESETS[2]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert scene.preset is PRESETS[0]
        assert scene.method == ImportanceMethod.PERMUTATION
        assert scene.show_correlation_groups is True
        assert scene.show_leakage_warning is True
    finally:
        pygame.quit()


def test_feature_importance_scene_detects_leakage_and_unstable_ranking(monkeypatch) -> None:
    """Scenario diagnostics should flag leakage and instability."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_importance_lab_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        assert scene.top_feature().name == "future_status"
        assert scene._diagnosis_key() == "leakage"
        assert "Investigate" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l))
        assert scene._diagnosis_key() != "leakage"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene._diagnosis_key() == "unstable"
    finally:
        pygame.quit()


def test_feature_importance_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize guidance while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_feature_importance_lab_scene(context)

        assert scene._diagnosis_label() == "wspólny sygnał"
        assert "Skorelowane cechy" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        assert scene._diagnosis_label() == "możliwy leakage"
    finally:
        pygame.quit()


def test_feature_importance_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The feature importance preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_feature_importance_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
