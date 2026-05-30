"""Tests for unified app shell settings."""

from interactive_ml_labs import AppContext, AppSettings
from interactive_ml_labs.settings import DEFAULT_RESOLUTION


def test_default_settings_are_in_memory_shell_defaults() -> None:
    """Default settings should match the first shell slice."""
    settings = AppSettings()

    assert settings.language == "en"
    assert settings.resolution == DEFAULT_RESOLUTION
    assert settings.sound_enabled is False


def test_app_context_tracks_current_navigation() -> None:
    """App context should keep selected level and demo id."""
    context = AppContext()
    context.current_level = 2
    context.selected_demo_id = "boosting_mistake_lab"

    assert context.current_level == 2
    assert context.selected_demo_id == "boosting_mistake_lab"
