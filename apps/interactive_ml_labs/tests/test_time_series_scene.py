"""Tests for the native Time Series Forecasting Lab scene."""

import pygame
from interactive_ml_labs.display import DEFAULT_RESOLUTION
from interactive_ml_labs.scene import FixedSizeScene, SceneCommandKind
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.time_series_scene import (
    COMPARE_FORECAST_CHOICES_TASK_ID,
    HORIZONS,
    INSPECT_FORECAST_RISK_TASK_ID,
    PRESETS,
    TIME_SERIES_LESSON_ID,
    ForecastModel,
    TimeSeriesForecastingLabScene,
    create_time_series_forecasting_lab_scene,
)


def test_time_series_scene_exposes_fixed_scene_contract(monkeypatch) -> None:
    """Time Series Forecasting scene should render into the stable shell surface."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_time_series_forecasting_lab_scene(AppContext())

        assert isinstance(scene, TimeSeriesForecastingLabScene)
        assert isinstance(scene, FixedSizeScene)
        assert scene.fixed_scene_size == DEFAULT_RESOLUTION
    finally:
        pygame.quit()


def test_time_series_scene_switches_model_horizon_and_resets(monkeypatch) -> None:
    """Keyboard controls should update the forecasting setup."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_time_series_forecasting_lab_scene(AppContext())

        assert scene.preset is PRESETS[0]
        assert scene.model == ForecastModel.TREND_SEASONAL
        assert scene.horizon == 6
        assert len(scene.forecast_values()) == scene.horizon

        command = scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        assert command.kind == SceneCommandKind.NONE
        assert scene.model == ForecastModel.NAIVE

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_EQUALS))
        assert scene.horizon == HORIZONS[-1]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_3))
        assert scene.preset is PRESETS[2]

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_u))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))
        assert scene.show_uncertainty is False
        assert scene.show_residuals is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert scene.preset is PRESETS[0]
        assert scene.model == ForecastModel.TREND_SEASONAL
        assert scene.horizon == 6
        assert scene.show_uncertainty is True
        assert scene.show_residuals is True
    finally:
        pygame.quit()


def test_time_series_scene_reports_forecast_error_metrics(monkeypatch) -> None:
    """Forecast error metrics should be positive and internally consistent."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_time_series_forecasting_lab_scene(AppContext())
        residuals = scene.residuals()

        assert len(residuals) == scene.horizon
        assert scene.mae() > 0
        assert scene.rmse() >= scene.mae()
        assert abs(scene.bias()) <= max(abs(value) for value in residuals)
        assert scene.uncertainty_width() > 0
    finally:
        pygame.quit()


def test_time_series_scene_localizes_polish_labels(monkeypatch) -> None:
    """Polish mode should keep ML terms while localizing natural UI copy."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        context.settings.language = "pl"
        scene = create_time_series_forecasting_lab_scene(context)

        assert scene.preset.name_for_language("pl") == "Sezonowy popyt"
        assert "sezonowość" in scene._model_label()
        assert "forecast" in scene._active_takeaway()
        assert scene._diagnosis_label() in {
            "forecast z bias",
            "szerokie residuals",
            "użyteczny forecast",
        }
    finally:
        pygame.quit()


def test_time_series_scene_reports_guided_lesson_progress(monkeypatch) -> None:
    """Guided mode should complete forecasting tasks from real interactions."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext(selected_lesson_id=TIME_SERIES_LESSON_ID)
        scene = create_time_series_forecasting_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))

        progress = context.progress.lessons[TIME_SERIES_LESSON_ID]
        assert progress.completed_task_ids == {COMPARE_FORECAST_CHOICES_TASK_ID}
        assert progress.completed is False

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))

        progress = context.progress.lessons[TIME_SERIES_LESSON_ID]
        assert progress.completed_task_ids == {
            COMPARE_FORECAST_CHOICES_TASK_ID,
            INSPECT_FORECAST_RISK_TASK_ID,
        }
        assert progress.completed is True
    finally:
        pygame.quit()


def test_time_series_scene_standalone_does_not_mutate_guided_progress(monkeypatch) -> None:
    """Standalone Time Series interactions should not complete guided lessons."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        context = AppContext()
        scene = create_time_series_forecasting_lab_scene(context)

        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        scene.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_e))

        assert context.progress.lessons == {}
    finally:
        pygame.quit()


def test_time_series_scene_renders_to_shell_surface(monkeypatch) -> None:
    """The forecasting lab should draw a non-empty frame."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        scene = create_time_series_forecasting_lab_scene(AppContext())
        surface = pygame.Surface(DEFAULT_RESOLUTION)

        scene.render(surface)

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
