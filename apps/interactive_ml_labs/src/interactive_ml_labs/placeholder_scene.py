"""Placeholder scene used before real demos are integrated."""

from __future__ import annotations

from dataclasses import dataclass

from interactive_ml_labs.manifest import DemoManifest
from interactive_ml_labs.settings import AppContext


@dataclass(slots=True)
class PlaceholderDemoScene:
    """Non-interactive scene that represents a not-yet-integrated demo."""

    context: AppContext
    manifest: DemoManifest

    def handle_event(self, event: object) -> None:
        """Handle one input event."""

    def update(self, dt: float) -> None:
        """Advance scene state."""

    def render(self, surface: object) -> None:
        """Draw the scene.

        The unified shell owns placeholder rendering for now. This method exists
        to satisfy the scene contract before real demo adapters are introduced.
        """
