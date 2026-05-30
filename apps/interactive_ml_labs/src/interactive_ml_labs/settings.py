"""Application settings and context for the unified shell."""

from dataclasses import dataclass, field

DEFAULT_RESOLUTION: tuple[int, int] = (1280, 720)


@dataclass(slots=True)
class AppSettings:
    """Mutable in-memory shell settings."""

    language: str = "en"
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    sound_enabled: bool = False


@dataclass(slots=True)
class AppContext:
    """Shared app context passed to shell screens and future demo scenes."""

    settings: AppSettings = field(default_factory=AppSettings)
    current_level: int | None = None
    selected_demo_id: str | None = None
    theme: str = "default"
