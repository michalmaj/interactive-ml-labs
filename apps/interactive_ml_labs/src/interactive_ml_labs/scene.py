"""Scene contracts and navigation helpers for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class SceneCommandKind(StrEnum):
    """Actions a scene may request from the shell."""

    NONE = "none"
    PAUSE = "pause"
    BACK_TO_DEMOS = "back_to_demos"
    RESTART = "restart"
    QUIT = "quit"


@dataclass(frozen=True, slots=True)
class SceneCommand:
    """Navigation command emitted by a scene."""

    kind: SceneCommandKind = SceneCommandKind.NONE

    @classmethod
    def none(cls) -> SceneCommand:
        """Return an empty command."""
        return cls(SceneCommandKind.NONE)

    @classmethod
    def pause(cls) -> SceneCommand:
        """Request the shell pause menu."""
        return cls(SceneCommandKind.PAUSE)

    @classmethod
    def back_to_demos(cls) -> SceneCommand:
        """Request navigation back to demo selection."""
        return cls(SceneCommandKind.BACK_TO_DEMOS)

    @classmethod
    def restart(cls) -> SceneCommand:
        """Request demo restart."""
        return cls(SceneCommandKind.RESTART)

    @classmethod
    def quit(cls) -> SceneCommand:
        """Request app quit."""
        return cls(SceneCommandKind.QUIT)


class Scene(Protocol):
    """Minimal scene protocol used by demo adapters."""

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""

    def render(self, surface: object) -> None:
        """Draw the scene."""


class SceneManager:
    """Small stack-based scene manager for demo scenes."""

    def __init__(self) -> None:
        """Create an empty scene stack."""
        self._stack: list[Scene] = []

    @property
    def current(self) -> Scene | None:
        """Return the active scene, if any."""
        if not self._stack:
            return None

        return self._stack[-1]

    def push(self, scene: Scene) -> None:
        """Push a scene on top of the stack."""
        self._stack.append(scene)

    def replace(self, scene: Scene) -> None:
        """Replace the current scene with a new scene."""
        if self._stack:
            self._stack[-1] = scene
            return

        self._stack.append(scene)

    def pop(self) -> Scene | None:
        """Pop and return the current scene."""
        if not self._stack:
            return None

        return self._stack.pop()

    def clear(self) -> None:
        """Remove all scenes."""
        self._stack.clear()

    def __len__(self) -> int:
        """Return scene stack depth."""
        return len(self._stack)
