"""Tests for the native Model Monitoring Drift Lab skeleton scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.monitoring_scene import (
    DEFAULT_THRESHOLD_INDEX,
    PRESETS,
    ModelMonitoringDriftScene,
    create_model_monitoring_drift_scene,
)
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext


def test_monitoring_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Model Monitoring scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())

        assert isinstance(scene, ModelMonitoringDriftScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_monitoring_scene_handles_events_without_navigation(monkeypatch) -> None:
    """Scene controls should update state without requesting navigation."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))
        update_command = scene.update(0.16)

        assert command.kind == SceneCommandKind.NONE
        assert update_command.kind == SceneCommandKind.NONE
        assert scene.preset_index == 1
        assert scene.preset is PRESETS[1]
    finally:
        pygame.quit()


def test_monitoring_scene_switches_signal_threshold_and_resets(monkeypatch) -> None:
    """D/M, threshold keys, and R should update monitoring state."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))

        assert scene.signal == "metric drift"
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX + 1
        assert scene.preset_index == 2

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))

        assert scene.signal == "data drift"
        assert scene.threshold_index == DEFAULT_THRESHOLD_INDEX
        assert scene.preset_index == 0
    finally:
        pygame.quit()


def test_monitoring_scene_reports_window_metrics_and_alerts(monkeypatch) -> None:
    """Window means and alerts should reflect baseline vs current drift."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())

        assert scene._window_mean(-1) >= scene._window_mean(0)
        assert scene._drift_gap() < scene.threshold
        assert scene._alert_active() is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_2))

        assert scene._drift_gap() >= scene.threshold
        assert scene._alert_active() is True
        assert "Alert" in scene._active_takeaway()
    finally:
        pygame.quit()


def test_monitoring_scene_localizes_preset_copy(monkeypatch) -> None:
    """Preset copy should use the global language from AppContext."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_model_monitoring_drift_scene(context)

        assert scene.preset.name_for_language("pl") == "Stable baseline"
        assert "baseline" in scene.preset.summary_for_language("pl")
    finally:
        pygame.quit()


def test_monitoring_scene_plot_rect_stays_inside_panel(monkeypatch) -> None:
    """Timeline plot should leave room for labels and summary copy."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())
        panel_rect = pygame.Rect(58, 132, 700, 474)
        plot_rect = scene._timeline_plot_rect(panel_rect)

        assert plot_rect.top >= panel_rect.top + 80
        assert plot_rect.bottom <= panel_rect.bottom - 76
    finally:
        pygame.quit()


def test_monitoring_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The monitoring preview should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_model_monitoring_drift_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
