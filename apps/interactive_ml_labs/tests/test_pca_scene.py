"""Tests for the native PCA Lab skeleton scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.pca_scene import (
    ANGLE_STEP_DEGREES,
    DEFAULT_PROJECTION_ANGLE_DEGREES,
    PCALabScene,
    create_pca_lab_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_pca_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """PCA Lab should be ready for shell-side scaling."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())

        assert isinstance(scene, PCALabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_pca_scene_handles_events_without_navigation(monkeypatch) -> None:
    """Projection controls should update the scene without requesting navigation."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
        update_command = scene.update(0.16)

        assert command.kind == SceneCommandKind.NONE
        assert update_command.kind == SceneCommandKind.NONE
        assert scene.projection_angle_degrees == (
            DEFAULT_PROJECTION_ANGLE_DEGREES + ANGLE_STEP_DEGREES
        )
    finally:
        pygame.quit()


def test_pca_scene_projection_controls_change_variance(monkeypatch) -> None:
    """Rotating the projection should change the explained variance."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        initial_variance = scene._explained_variance_ratio()

        for _ in range(9):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert scene.projection_angle_degrees != DEFAULT_PROJECTION_ANGLE_DEGREES
        assert scene._explained_variance_ratio() != initial_variance

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.projection_angle_degrees == DEFAULT_PROJECTION_ANGLE_DEGREES
        assert scene._explained_variance_ratio() == initial_variance
    finally:
        pygame.quit()


def test_pca_scene_status_rows_report_variance(monkeypatch) -> None:
    """The status panel should expose angle, kept variance, and lost variance."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_pca_lab_scene(context)
        rows = dict(scene._status_rows())

        assert rows["kąt"] == f"{DEFAULT_PROJECTION_ANGLE_DEGREES}°"
        assert rows["zachowana wariancja"].endswith("%")
        assert rows["utracona wariancja"].endswith("%")
    finally:
        pygame.quit()


def test_pca_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The PCA preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
