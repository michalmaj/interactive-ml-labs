"""Application settings and context for the unified shell."""

from dataclasses import dataclass, field

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size


@dataclass(slots=True)
class AppSettings:
    """Mutable in-memory shell settings."""

    language: str = "en"
    resolution: Size = DEFAULT_RESOLUTION
    adaptive_window_enabled: bool = False
    fullscreen_enabled: bool = False
    sound_enabled: bool = False


@dataclass(slots=True)
class AppContext:
    """Shared app context passed to shell screens and future demo scenes."""

    settings: AppSettings = field(default_factory=AppSettings)
    current_level: int | None = None
    selected_demo_id: str | None = None
    theme: str = "default"
