"""Placeholder scene used before real demos are integrated."""

from __future__ import annotations

from dataclasses import dataclass

from interactive_ml_labs.manifest import DemoManifest
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext


@dataclass(slots=True)
class PlaceholderDemoScene:
    """Non-interactive scene that represents a not-yet-integrated demo."""

    context: AppContext
    manifest: DemoManifest

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        _ = event
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the scene.

        The unified shell owns placeholder rendering for now. This method exists
        to satisfy the scene contract before real demo adapters are introduced.
        """
