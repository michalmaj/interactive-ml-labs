"""Tests for unified app shell settings."""

import json

from interactive_ml_labs import AppContext, AppProgress, AppSettings
from interactive_ml_labs.display import (
    BOOSTING_FIXED_SCENE_SIZE,
    DEFAULT_RESOLUTION,
    SCREEN_MARGIN,
    WINDOWED_RESOLUTIONS,
    center_rect,
    choose_adaptive_window_size,
    scale_rect_to_fit,
)
from interactive_ml_labs.settings import (
    load_app_settings,
    save_app_settings,
    settings_from_json,
    settings_to_json,
)


def test_default_settings_are_in_memory_shell_defaults() -> None:
    """Default settings should match the first shell slice."""
    settings = AppSettings()

    assert settings.language == "en"
    assert settings.resolution == DEFAULT_RESOLUTION
    assert settings.adaptive_window_enabled is False
    assert settings.fixed_scene_scaling_enabled is True
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


def test_scale_rect_to_fit_preserves_aspect_ratio_in_smaller_window() -> None:
    """Fixed scenes should be letterboxed when scaled into a smaller window."""
    assert scale_rect_to_fit((1280, 720), BOOSTING_FIXED_SCENE_SIZE) == (
        31,
        0,
        1218,
        720,
    )


def test_scale_rect_to_fit_can_upscale_into_larger_window() -> None:
    """Fixed scenes should also scale up for roomier windows."""
    assert scale_rect_to_fit((1600, 900), BOOSTING_FIXED_SCENE_SIZE) == (
        38,
        0,
        1523,
        900,
    )


def test_scale_rect_to_fit_handles_invalid_content_size() -> None:
    """Invalid logical scene sizes should produce an empty rect."""
    assert scale_rect_to_fit((1280, 720), (0, 780)) == (0, 0, 0, 0)


def test_app_context_tracks_current_navigation() -> None:
    """App context should keep selected level and demo id."""
    context = AppContext()
    context.current_level = 2
    context.selected_demo_id = "boosting_mistake_lab"

    assert context.current_level == 2
    assert context.selected_demo_id == "boosting_mistake_lab"


def test_app_context_carries_learning_progress() -> None:
    """App context should carry lesson progress for shell and future scenes."""
    context = AppContext()

    context.progress.mark_started("lesson_one")

    assert isinstance(context.progress, AppProgress)
    assert context.progress.lessons["lesson_one"].started is True


def test_settings_serialization_persists_user_preferences_only() -> None:
    """Persistent settings should store user intent, not computed window size."""
    settings = AppSettings(
        language="pl",
        resolution=(1920, 1080),
        fullscreen_enabled=True,
        adaptive_window_enabled=True,
        fixed_scene_scaling_enabled=False,
        sound_enabled=True,
    )

    data = settings_to_json(settings)

    assert data == {
        "version": 1,
        "language": "pl",
        "fullscreen_enabled": True,
        "adaptive_window_enabled": True,
        "fixed_scene_scaling_enabled": False,
    }


def test_settings_from_json_uses_defaults_for_invalid_values() -> None:
    """Malformed values should not break app startup."""
    settings = settings_from_json(
        {
            "language": "de",
            "fullscreen_enabled": "yes",
            "adaptive_window_enabled": True,
            "fixed_scene_scaling_enabled": False,
        },
    )

    assert settings.language == "en"
    assert settings.fullscreen_enabled is False
    assert settings.adaptive_window_enabled is True
    assert settings.fixed_scene_scaling_enabled is False


def test_load_app_settings_returns_defaults_when_file_is_missing(tmp_path) -> None:
    """Missing settings files should keep first-run defaults."""
    settings = load_app_settings(tmp_path / "missing" / "settings.json")

    assert settings == AppSettings()


def test_save_and_load_app_settings_round_trip(tmp_path) -> None:
    """Settings should round-trip through the per-user JSON file."""
    settings_path = tmp_path / "interactive-ml-labs" / "settings.json"
    settings = AppSettings(
        language="pl",
        fullscreen_enabled=True,
        adaptive_window_enabled=True,
        fixed_scene_scaling_enabled=False,
    )

    save_app_settings(settings, settings_path)
    loaded_settings = load_app_settings(settings_path)

    assert loaded_settings.language == "pl"
    assert loaded_settings.fullscreen_enabled is True
    assert loaded_settings.adaptive_window_enabled is True
    assert loaded_settings.fixed_scene_scaling_enabled is False
    assert loaded_settings.resolution == DEFAULT_RESOLUTION
    assert json.loads(settings_path.read_text(encoding="utf-8")) == settings_to_json(settings)


def test_load_app_settings_returns_defaults_for_invalid_json(tmp_path) -> None:
    """A corrupt settings file should be ignored."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("{", encoding="utf-8")

    assert load_app_settings(settings_path) == AppSettings()
