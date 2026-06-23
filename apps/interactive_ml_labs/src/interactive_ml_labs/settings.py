"""Application settings and context for the unified shell."""

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.progress import AppProgress

SETTINGS_FILE_NAME = "settings.json"
SETTINGS_SCHEMA_VERSION = 1


@dataclass(slots=True)
class AppSettings:
    """Mutable shell settings."""

    language: str = "en"
    resolution: Size = DEFAULT_RESOLUTION
    adaptive_window_enabled: bool = False
    fixed_scene_scaling_enabled: bool = True
    fullscreen_enabled: bool = False
    sound_enabled: bool = False


@dataclass(slots=True)
class AppContext:
    """Shared app context passed to shell screens and future demo scenes."""

    settings: AppSettings = field(default_factory=AppSettings)
    progress: AppProgress = field(default_factory=AppProgress)
    current_level: int | None = None
    selected_demo_id: str | None = None
    theme: str = "default"


def default_settings_path() -> Path:
    """Return the per-user settings path for the shell."""
    if os.name == "nt":
        config_root = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif xdg_config_home := os.environ.get("XDG_CONFIG_HOME"):
        config_root = Path(xdg_config_home)
    elif sys.platform == "darwin":
        config_root = Path.home() / "Library" / "Application Support"
    else:
        config_root = Path.home() / ".config"

    return config_root / "interactive-ml-labs" / SETTINGS_FILE_NAME


def settings_to_json(settings: AppSettings) -> dict[str, object]:
    """Serialize persistent settings to a small JSON-friendly mapping."""
    return {
        "version": SETTINGS_SCHEMA_VERSION,
        "language": settings.language,
        "fullscreen_enabled": settings.fullscreen_enabled,
        "adaptive_window_enabled": settings.adaptive_window_enabled,
        "fixed_scene_scaling_enabled": settings.fixed_scene_scaling_enabled,
    }


def settings_from_json(data: object) -> AppSettings:
    """Build settings from JSON data, falling back for unknown or malformed values."""
    settings = AppSettings()
    if not isinstance(data, dict):
        return settings

    language = data.get("language")
    if language in {"en", "pl"}:
        settings.language = language

    for field_name in (
        "fullscreen_enabled",
        "adaptive_window_enabled",
        "fixed_scene_scaling_enabled",
    ):
        value = data.get(field_name)
        if isinstance(value, bool):
            setattr(settings, field_name, value)

    return settings


def load_app_settings(path: Path | None = None) -> AppSettings:
    """Load persisted settings, returning defaults when the file is absent or invalid."""
    settings_path = path or default_settings_path()
    try:
        with settings_path.open(encoding="utf-8") as file:
            return settings_from_json(json.load(file))
    except (OSError, json.JSONDecodeError):
        return AppSettings()


def save_app_settings(settings: AppSettings, path: Path | None = None) -> None:
    """Persist settings best-effort without interrupting the app on write errors."""
    settings_path = path or default_settings_path()
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with settings_path.open("w", encoding="utf-8") as file:
            json.dump(settings_to_json(settings), file, ensure_ascii=False, indent=2)
            file.write("\n")
    except OSError:
        return
