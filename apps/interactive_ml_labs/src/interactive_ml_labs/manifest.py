"""Demo manifest models for the unified app shell."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from interactive_ml_labs.scene import Scene


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


@dataclass(frozen=True, slots=True)
class TheorySection:
    """Short generated theory block for the in-app lesson screen."""

    title: LocalizedText
    body: tuple[LocalizedText, ...]


@dataclass(frozen=True, slots=True)
class DemoTheory:
    """Compact in-app theory content for one demo."""

    sections: tuple[TheorySection, ...]


@dataclass(frozen=True, slots=True)
class LevelManifest:
    """Metadata for one learning level."""

    number: int
    title: LocalizedText
    summary: LocalizedText


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
    theory: DemoTheory | None = None
