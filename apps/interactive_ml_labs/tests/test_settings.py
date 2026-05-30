"""Tests for unified app shell settings."""

from interactive_ml_labs import AppContext, AppSettings
from interactive_ml_labs.display import (
    BOOSTING_FIXED_SCENE_SIZE,
    DEFAULT_RESOLUTION,
    WINDOWED_RESOLUTIONS,
    center_rect,
)


def test_default_settings_are_in_memory_shell_defaults() -> None:
    """Default settings should match the first shell slice."""
    settings = AppSettings()

    assert settings.language == "en"
    assert settings.resolution == DEFAULT_RESOLUTION
    assert settings.fullscreen_enabled is False
    assert settings.sound_enabled is False


def test_default_resolution_is_larger_windowed_app_size() -> None:
    """Default app resolution should be the larger guided-app window size."""
    assert DEFAULT_RESOLUTION == (1600, 900)
    assert DEFAULT_RESOLUTION in WINDOWED_RESOLUTIONS


def test_boosting_fixed_scene_can_be_centered_in_default_resolution() -> None:
    """Fixed Boosting scene size should center cleanly in the default window."""
    assert center_rect(DEFAULT_RESOLUTION, BOOSTING_FIXED_SCENE_SIZE) == (
        140,
        60,
        1320,
        780,
    )


def test_center_rect_does_not_return_negative_offsets() -> None:
    """Centered content should clamp offsets when content is larger than container."""
    assert center_rect((640, 360), (1320, 780)) == (0, 0, 1320, 780)


def test_app_context_tracks_current_navigation() -> None:
    """App context should keep selected level and demo id."""
    context = AppContext()
    context.current_level = 2
    context.selected_demo_id = "boosting_mistake_lab"

    assert context.current_level == 2
    assert context.selected_demo_id == "boosting_mistake_lab"
