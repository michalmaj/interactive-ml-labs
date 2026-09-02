"""Shared pytest defaults for the workspace."""

from __future__ import annotations

import os


def pytest_configure() -> None:
    """Keep Pygame tests headless unless a caller selects another video driver."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
