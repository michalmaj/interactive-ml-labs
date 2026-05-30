"""Demo manifest models for the unified app shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class LocalizedText:
    """Small two-language text value."""

    en: str
    pl: str

    def for_language(self, language: str) -> str:
        """Return localized text, falling back to English."""
        if language == "pl":
            return self.pl

        return self.en


@dataclass(frozen=True, slots=True)
class ControlBinding:
    """Human-readable control description."""

    key: str
    action: LocalizedText


class Scene(Protocol):
    """Minimal scene protocol used by the shell."""

    def handle_event(self, event: object) -> None:
        """Handle one input event."""

    def update(self, dt: float) -> None:
        """Advance scene state."""

    def render(self, surface: object) -> None:
        """Draw the scene."""


@dataclass(frozen=True, slots=True)
class DemoManifest:
    """Metadata and scene factory for one demo."""

    id: str
    level: int
    title: LocalizedText
    summary: LocalizedText
    objectives: tuple[LocalizedText, ...]
    controls: tuple[ControlBinding, ...]
    create_scene: Callable[[object], Scene] | None = None
    difficulty: LocalizedText | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
