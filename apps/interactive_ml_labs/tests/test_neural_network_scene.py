"""Tests for the native Neural Network Playground scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.neural_network_scene import (
    ACTIVATIONS,
    BIAS_STEP,
    PRESETS,
    WEIGHT_STEP,
    NeuralNetworkPlaygroundScene,
    create_neural_network_playground_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_neural_network_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Neural network scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_neural_network_playground_scene(AppContext())

        assert isinstance(scene, NeuralNetworkPlaygroundScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_neural_network_scene_updates_state_and_resets(monkeypatch) -> None:
    """Preset, activation, weight, bias, and reset controls should update state."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_neural_network_playground_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.activation_name == ACTIVATIONS[0]
        assert scene.weight_scale == 1.0
        assert scene.hidden_bias == 0.0

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert command.kind == SceneCommandKind.NONE
        assert scene.preset is PRESETS[1]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        assert scene.activation_name == ACTIVATIONS[1]
        assert scene.weight_scale == 1.0 + WEIGHT_STEP
        assert scene.hidden_bias == BIAS_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.activation_name == ACTIVATIONS[0]
        assert scene.weight_scale == 1.0
        assert scene.hidden_bias == 0.0
    finally:
        pygame.quit()


def test_neural_network_scene_forward_pass_and_localization(monkeypatch) -> None:
    """Forward pass values should stay valid and labels should localize."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_neural_network_playground_scene(context)
        forward = scene._forward()

        assert 0.0 < forward.probability < 1.0
        assert forward.loss > 0.0
        assert scene._diagnosis_label() in {"dobra strona", "zła strona"}

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a))

        assert scene.activation_name == "ReLU"
        assert (
            "hidden unit" in scene._active_takeaway() or "Forward pass" in scene._active_takeaway()
        )
    finally:
        pygame.quit()


def test_neural_network_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The neural network preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_neural_network_playground_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
