"""Tests for the native SVM Margin Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.svm_margin_scene import (
    ANGLE_STEP,
    OFFSET_STEP,
    PRESETS,
    SVMMarginLabScene,
    create_svm_margin_lab_scene,
)


def test_svm_margin_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """SVM margin scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_svm_margin_lab_scene(AppContext())

        assert isinstance(scene, SVMMarginLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_svm_margin_scene_updates_boundary_and_resets(monkeypatch) -> None:
    """Arrow keys, preset keys, F, and R should update the margin preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_svm_margin_lab_scene(AppContext())
        start_angle = scene.angle
        start_offset = scene.offset

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert command.kind == SceneCommandKind.NONE
        assert scene.angle == start_angle + ANGLE_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        assert scene.offset == start_offset + OFFSET_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene.angle == PRESETS[0].best_angle
        assert scene.offset == PRESETS[0].best_offset
        assert scene._diagnosis_key() == "wide"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.preset is PRESETS[1]
        assert scene.angle == PRESETS[1].start_angle
        assert scene.offset == PRESETS[1].start_offset

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.angle == PRESETS[0].start_angle
        assert scene.offset == PRESETS[0].start_offset
    finally:
        pygame.quit()


def test_svm_margin_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize diagnosis labels while keeping SVM terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_svm_margin_lab_scene(context)

        assert scene._diagnosis_label() in {
            "błędy klasyfikacji",
            "szeroki margin",
            "wąski margin",
        }

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene._diagnosis_label() == "szeroki margin"
        assert "Support vectors" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_svm_margin_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The SVM margin preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_svm_margin_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
