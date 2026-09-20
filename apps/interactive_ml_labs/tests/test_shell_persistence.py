"""Tests for unified shell persistence facade."""

from interactive_ml_labs.progress import AppProgress, load_app_progress
from interactive_ml_labs.settings import AppSettings, load_app_settings
from interactive_ml_labs.shell_persistence import ShellPersistence, default_progress_path_for


def test_default_progress_path_for_keeps_progress_next_to_settings(tmp_path) -> None:
    """Explicit settings files should pair with a sibling progress file."""
    assert default_progress_path_for(tmp_path / "settings.json") == tmp_path / "progress.json"
    assert default_progress_path_for(None) is None


def test_shell_persistence_loads_settings_and_progress_from_explicit_paths(tmp_path) -> None:
    """The facade should load app state from explicit test paths."""
    settings_path = tmp_path / "settings.json"
    progress_path = tmp_path / "progress.json"
    settings = AppSettings(language="pl", fullscreen_enabled=True)
    progress = AppProgress()
    progress.mark_started("lesson-one")

    persistence, _context = ShellPersistence.create(
        settings_path=settings_path,
        progress_path=progress_path,
    )
    persistence.save_settings(settings)
    persistence.save_progress(progress)
    loaded_persistence, loaded_context = ShellPersistence.create(
        settings_path=settings_path,
        progress_path=progress_path,
    )

    assert loaded_persistence.settings_path == settings_path
    assert loaded_persistence.progress_path == progress_path
    assert loaded_context.settings.language == "pl"
    assert loaded_context.settings.fullscreen_enabled is True
    assert loaded_context.progress.lessons["lesson-one"].started is True


def test_shell_persistence_keeps_progress_in_memory_for_explicit_settings(tmp_path) -> None:
    """Explicit in-memory settings should not enable progress files by accident."""
    persistence, context = ShellPersistence.create(
        settings=AppSettings(resolution=(640, 360)),
    )
    persistence.progress_path = tmp_path / "progress.json"

    context.progress.mark_started("lesson-one")
    persistence.save_progress(context.progress)

    assert not persistence.progress_path.exists()


def test_shell_persistence_skips_unchanged_progress_writes(tmp_path) -> None:
    """Progress should be written only when its revision changes."""
    progress_path = tmp_path / "progress.json"
    progress = AppProgress()
    persistence, _ = ShellPersistence.create(progress_path=progress_path)

    persistence.save_progress(progress)
    assert not progress_path.exists()

    progress.mark_started("lesson-one")
    persistence.save_progress(progress)

    assert load_app_progress(progress_path).lessons["lesson-one"].started is True


def test_shell_persistence_saves_settings_when_enabled(tmp_path) -> None:
    """Settings writes should still use the existing JSON format."""
    settings_path = tmp_path / "settings.json"
    persistence, _ = ShellPersistence.create(settings_path=settings_path)

    persistence.save_settings(AppSettings(language="pl"))

    assert load_app_settings(settings_path).language == "pl"
