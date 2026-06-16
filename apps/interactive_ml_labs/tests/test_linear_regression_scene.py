"""Tests for the native Linear Regression Line Fit Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.linear_regression_scene import (
    INTERCEPT_STEP,
    PRESETS,
    SLOPE_STEP,
    LinearRegressionLineFitLabScene,
    create_linear_regression_line_fit_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_linear_regression_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Linear regression scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_linear_regression_line_fit_lab_scene(AppContext())

        assert isinstance(scene, LinearRegressionLineFitLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_linear_regression_scene_updates_line_and_resets(monkeypatch) -> None:
    """Arrow keys, preset keys, F, and R should update the line fit preview."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_linear_regression_line_fit_lab_scene(AppContext())
        start_slope = scene.slope
        start_intercept = scene.intercept
        start_mse = scene._mse()

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert command.kind == SceneCommandKind.NONE
        assert scene.slope == start_slope + SLOPE_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))

        assert scene.intercept == start_intercept + INTERCEPT_STEP

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene._mse() < start_mse
        assert scene._diagnosis_key() == "close"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene.preset is PRESETS[1]
        assert scene.slope == PRESETS[1].start_slope
        assert scene.intercept == PRESETS[1].start_intercept

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.preset is PRESETS[0]
        assert scene.slope == PRESETS[0].start_slope
        assert scene.intercept == PRESETS[0].start_intercept
    finally:
        pygame.quit()


def test_linear_regression_scene_localizes_labels(monkeypatch) -> None:
    """Polish mode should localize diagnosis text while keeping ML terms."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_linear_regression_line_fit_lab_scene(context)

        assert scene._diagnosis_label() == "popraw slope"

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene._diagnosis_label() == "blisko best fit"
        assert "Residuals są zbalansowane" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_linear_regression_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The line fit preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_linear_regression_line_fit_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
