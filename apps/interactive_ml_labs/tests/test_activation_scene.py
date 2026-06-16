"""Tests for the native Activation Functions Lab scene."""

import pygame
from interactive_ml_labs.activation_scene import (
    ACTIVATIONS,
    INPUT_STEP,
    ActivationFunctionsLabScene,
    create_activation_functions_lab_scene,
)
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_activation_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Activation scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_activation_functions_lab_scene(AppContext())

        assert isinstance(scene, ActivationFunctionsLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_activation_scene_updates_activation_input_and_resets(monkeypatch) -> None:
    """Number keys, arrows, 0, and R should update the activation preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_activation_functions_lab_scene(AppContext())

        assert scene.activation is ACTIVATIONS[0]
        assert scene.input_value == 0.0

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert command.kind == SceneCommandKind.NONE
        assert scene.input_value == INPUT_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.activation is ACTIVATIONS[2]
        assert scene._range_label() == "0..inf"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_0))

        assert scene.input_value == 0.0

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.activation is ACTIVATIONS[0]
        assert scene.input_value == 0.0
    finally:
        pygame.quit()


def test_activation_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize diagnosis labels while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_activation_functions_lab_scene(context)

        assert scene._diagnosis_label() == "gradient płynie"

        for _ in range(20):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert scene._diagnosis_label() == "saturacja"
        assert "local gradient" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_activation_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The activation preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_activation_functions_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
