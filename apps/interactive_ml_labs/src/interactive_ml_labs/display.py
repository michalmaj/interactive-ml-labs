"""Display helpers for the unified app shell."""

from __future__ import annotations

from typing import Final

type Size = tuple[int, int]
type RectTuple = tuple[int, int, int, int]

WINDOWED_RESOLUTIONS: Final[tuple[Size, ...]] = (
    (1280, 720),
    (1320, 780),
    (1600, 900),
    (1920, 1080),
)

DEFAULT_RESOLUTION: Final[Size] = (1280, 720)
BOOSTING_FIXED_SCENE_SIZE: Final[Size] = (1320, 780)


def center_rect(container_size: Size, content_size: Size) -> RectTuple:
    """Return a centered rect tuple for fixed-size content."""
    container_width, container_height = container_size
    content_width, content_height = content_size

    left = max(0, (container_width - content_width) // 2)
    top = max(0, (container_height - content_height) // 2)

    return (left, top, content_width, content_height)
