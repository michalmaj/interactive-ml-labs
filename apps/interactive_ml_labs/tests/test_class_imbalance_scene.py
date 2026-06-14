"""Tests for the native Class Imbalance Lab scene."""

import pygame
from interactive_ml_labs.class_imbalance_scene import (
    DEFAULT_THRESHOLD_INDEX,
    PRESETS,
    ClassImbalanceLabScene,
    create_class_imbalance_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_class_imbalance_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Class Imbalance scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_class_imbalance_lab_scene(AppContext())

        assert isinstance(scene, ClassImbalanceLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_class_imbalance_scene_updates_threshold_and_resets(monkeypatch) -> None:
    """Threshold keys, preset keys, and R should update the preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_class_imbalance_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX
        assert scene._threshold_label() == "50% (2/3)"
        assert scene._accuracy_label() == "95%"
        assert scene._precision_label() == "60%"
        assert scene._recall_label() == "55%"
        assert scene._diagnosis_key() == "accuracy_trap"

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

        assert command.kind == SceneCommandKind.NONE
        assert scene._threshold_label() == "30% (1/3)"
        assert scene._recall_label() == "80%"
        assert scene._diagnosis_key() == "wide_net"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset is PRESETS[2]
        assert scene._positive_share_label() == "18%"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX
    finally:
        pygame.quit()


def test_class_imbalance_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize operational labels while keeping metric terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_class_imbalance_lab_scene(context)

        assert scene._positive_label() == "fraud (6%)"
        assert scene._diagnosis_label() == "pułapka accuracy"
        assert "Accuracy wygląda wysoko" in scene._active_takeaway()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

        assert scene._diagnosis_label() == "szeroka sieć"
    finally:
        pygame.quit()


def test_class_imbalance_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The imbalance preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_class_imbalance_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
