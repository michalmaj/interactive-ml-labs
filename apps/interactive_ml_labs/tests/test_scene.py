"""Tests for scene commands and scene manager behavior."""

from __future__ import annotations

from dataclasses import dataclass

from interactive_ml_labs import SceneCommand, SceneCommandKind, SceneManager


@dataclass(slots=True)
class DummyScene:
    """Tiny scene double for manager tests."""

    name: str

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        _ = event
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the scene."""
        _ = surface


def test_scene_command_constructors() -> None:
    """Scene commands should expose stable command kinds."""
    assert SceneCommand.none().kind == SceneCommandKind.NONE
    assert SceneCommand.pause().kind == SceneCommandKind.PAUSE
    assert SceneCommand.back_to_demos().kind == SceneCommandKind.BACK_TO_DEMOS
    assert SceneCommand.restart().kind == SceneCommandKind.RESTART
    assert SceneCommand.quit().kind == SceneCommandKind.QUIT


def test_scene_manager_push_replace_pop_and_clear() -> None:
    """Scene manager should maintain a small stack of scenes."""
    manager = SceneManager()
    first = DummyScene("first")
    second = DummyScene("second")
    replacement = DummyScene("replacement")

    assert manager.current is None
    assert len(manager) == 0

    manager.push(first)
    manager.push(second)

    assert manager.current is second
    assert len(manager) == 2

    manager.replace(replacement)

    assert manager.current is replacement
    assert len(manager) == 2
    assert manager.pop() is replacement
    assert manager.current is first

    manager.clear()

    assert manager.current is None
    assert len(manager) == 0


def test_scene_manager_replace_on_empty_stack_pushes_scene() -> None:
    """Replacing an empty scene stack should install the scene."""
    manager = SceneManager()
    scene = DummyScene("replacement")

    manager.replace(scene)

    assert manager.current is scene
    assert len(manager) == 1
