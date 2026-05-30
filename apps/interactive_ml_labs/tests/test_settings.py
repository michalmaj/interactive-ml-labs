"""Tests for unified app shell settings."""

from interactive_ml_labs import AppContext, AppSettings
from interactive_ml_labs.display import (
    BOOSTING_FIXED_SCENE_SIZE,
    DEFAULT_RESOLUTION,
    SCREEN_MARGIN,
    WINDOWED_RESOLUTIONS,
    center_rect,
    choose_adaptive_window_size,
)


def test_default_settings_are_in_memory_shell_defaults() -> None:
    """Default settings should match the first shell slice."""
    settings = AppSettings()

    assert settings.language == "en"
    assert settings.resolution == DEFAULT_RESOLUTION
    assert settings.adaptive_window_enabled is False
    assert settings.fullscreen_enabled is False
    assert settings.sound_enabled is False


def test_default_resolution_is_safe_windowed_app_size() -> None:
    """Default app resolution should fit common laptop displays."""
    assert DEFAULT_RESOLUTION == (1280, 720)
    assert DEFAULT_RESOLUTION in WINDOWED_RESOLUTIONS


def test_adaptive_window_uses_largest_preset_that_fits_laptop_display() -> None:
    """Adaptive sizing should pick a larger window only when it fits safely."""
    assert choose_adaptive_window_size((1512, 982)) == (1320, 780)


def test_adaptive_window_uses_largest_preset_for_large_display() -> None:
    """Adaptive sizing should use the largest known preset on roomy displays."""
    assert choose_adaptive_window_size((2560, 1440)) == (1920, 1080)


def test_adaptive_window_falls_back_when_display_is_too_small() -> None:
    """Adaptive sizing should keep the safe default when no preset fits."""
    assert choose_adaptive_window_size((1024, 768)) == DEFAULT_RESOLUTION


def test_adaptive_window_accepts_custom_margin_and_fallback() -> None:
    """Adaptive sizing should stay usable for tests and future settings menus."""
    assert choose_adaptive_window_size(
        (1024, 768),
        presets=((800, 600), (900, 700)),
        fallback=(640, 360),
        margin=(SCREEN_MARGIN[0], 40),
    ) == (900, 700)


def test_boosting_fixed_scene_can_be_centered_in_larger_resolution() -> None:
    """Fixed Boosting scene size should center cleanly in a larger window."""
    assert center_rect((1600, 900), BOOSTING_FIXED_SCENE_SIZE) == (
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
