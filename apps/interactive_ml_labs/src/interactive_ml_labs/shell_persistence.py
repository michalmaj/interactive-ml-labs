"""Persistence facade for the unified app shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from interactive_ml_labs.progress import AppProgress, load_app_progress, save_app_progress
from interactive_ml_labs.settings import (
    AppContext,
    AppSettings,
    load_app_settings,
    save_app_settings,
)


def default_progress_path_for(settings_path: Path | None) -> Path | None:
    """Keep explicit test settings and progress files next to each other."""
    if settings_path is None:
        return None

    return settings_path.with_name("progress.json")


@dataclass(slots=True)
class ShellPersistence:
    """Small facade for shell-owned settings and progress persistence."""

    settings_path: Path | None
    progress_path: Path | None
    settings_enabled: bool
    progress_enabled: bool
    saved_progress_revision: int

    @classmethod
    def create(
        cls,
        *,
        settings: AppSettings | None = None,
        settings_path: Path | None = None,
        progress: AppProgress | None = None,
        progress_path: Path | None = None,
    ) -> tuple[ShellPersistence, AppContext]:
        """Create persistence state and load the initial app context."""
        resolved_progress_path = progress_path or default_progress_path_for(settings_path)
        persistence = cls(
            settings_path=settings_path,
            progress_path=resolved_progress_path,
            settings_enabled=settings is None or settings_path is not None,
            progress_enabled=progress is None
            and (settings is None or resolved_progress_path is not None),
            saved_progress_revision=0,
        )
        context = AppContext(
            settings=settings or load_app_settings(settings_path),
            progress=progress
            or persistence.load_initial_progress(
                explicit_settings=settings,
            ),
        )
        persistence.saved_progress_revision = context.progress.revision
        return persistence, context

    def load_initial_progress(
        self,
        *,
        explicit_settings: AppSettings | None,
    ) -> AppProgress:
        """Load progress only when the shell owns persistent app state."""
        if explicit_settings is not None and self.progress_path is None:
            return AppProgress()

        return load_app_progress(self.progress_path)

    def save_settings(self, settings: AppSettings) -> None:
        """Persist settings when the shell owns settings storage."""
        if self.settings_enabled:
            save_app_settings(settings, self.settings_path)

    def save_progress(self, progress: AppProgress) -> None:
        """Persist progress when the shell owns progress storage."""
        if progress.revision == self.saved_progress_revision:
            return

        if self.progress_enabled:
            save_app_progress(progress, self.progress_path)
        self.saved_progress_revision = progress.revision
