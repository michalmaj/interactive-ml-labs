"""Tests for the native PCA Lab skeleton scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.pca_scene import (
    ANGLE_STEP_DEGREES,
    DATA_PRESETS,
    DEFAULT_PROJECTION_ANGLE_DEGREES,
    MAX_NOISE_LEVEL,
    MIN_NOISE_LEVEL,
    PCALabScene,
    PCAProjectionMode,
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


def test_pca_scene_fit_mode_uses_best_variance_direction(monkeypatch) -> None:
    """F should switch to the fitted PCA direction."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        fitted_angle = scene._fitted_pca_angle_degrees()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene.projection_mode == PCAProjectionMode.FIT
        assert scene._active_projection_angle_degrees() == fitted_angle
        assert scene._explained_variance_ratio() >= 0.95

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))

        assert scene.projection_mode == PCAProjectionMode.MANUAL
    finally:
        pygame.quit()


def test_pca_scene_switches_dataset_presets(monkeypatch) -> None:
    """Number keys should switch PCA dataset presets."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        first_points = tuple(scene._points)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.preset_index == 2
        assert scene.preset is DATA_PRESETS[2]
        assert tuple(scene._points) != first_points
    finally:
        pygame.quit()


def test_pca_scene_noise_controls_update_dataset(monkeypatch) -> None:
    """Minus and equals should update noise within scene limits."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        first_points = tuple(scene._points)

        for _ in range(8):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.noise_level == MAX_NOISE_LEVEL
        assert tuple(scene._points) != first_points

        for _ in range(8):
            scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))
        assert scene.noise_level == MIN_NOISE_LEVEL
    finally:
        pygame.quit()


def test_pca_scene_new_sample_regenerates_points(monkeypatch) -> None:
    """N should regenerate the active PCA dataset sample."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        first_points = tuple(scene._points)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))

        assert scene.sample_seed == 12
        assert tuple(scene._points) != first_points
    finally:
        pygame.quit()


def test_pca_scene_toggles_residual_lines(monkeypatch) -> None:
    """C should show or hide reconstruction residual helper lines."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        assert scene.show_residuals is True

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.show_residuals is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_c))
        assert scene.show_residuals is True
    finally:
        pygame.quit()


def test_pca_scene_projects_points_to_active_axis(monkeypatch) -> None:
    """Point projections should lie on the active projection axis."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        point = scene._points[4]
        projected = scene._project_point_to_active_axis(point)
        mean_x, mean_y = scene._point_mean()
        direction_x, direction_y = scene._projection_direction()
        cross_product = (projected[0] - mean_x) * direction_y - (
            projected[1] - mean_y
        ) * direction_x

        assert abs(cross_product) < 0.000001
    finally:
        pygame.quit()


def test_pca_scene_manual_rotation_from_fit_mode(monkeypatch) -> None:
    """Left and Right should leave fit mode and continue from the PCA direction."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_pca_lab_scene(AppContext())
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        fitted_angle = scene._fitted_pca_angle_degrees()

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))

        assert scene.projection_mode == PCAProjectionMode.MANUAL
        assert scene.projection_angle_degrees == (fitted_angle + ANGLE_STEP_DEGREES) % 180
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

        assert rows["tryb"] == "manual"
        assert rows["dane"] == DATA_PRESETS[0].name_pl
        assert rows["noise"] == "1"
        assert rows["residuals"] == "wł."
        assert rows["kąt"] == f"{DEFAULT_PROJECTION_ANGLE_DEGREES}°"
        assert rows["zachowana wariancja"].endswith("%")
        assert rows["błąd rekonstrukcji"].endswith("%")

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        rows = dict(scene._status_rows())

        assert rows["tryb"] == "fit PCA"
    finally:
        pygame.quit()


def test_pca_scene_variance_panel_keeps_help_below_status(monkeypatch) -> None:
    """Explained variance help text should not overlap the status rows."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_pca_lab_scene(context)
        panel_rect = pygame.Rect(960, 138, 260, 420)
        status_bottom = (
            scene._explained_variance_status_start_y(panel_rect) + len(scene._status_rows()) * 18
        )
        help_y = scene._explained_variance_help_y(panel_rect)

        assert help_y >= status_bottom + 12
        assert help_y <= panel_rect.bottom - 56
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
