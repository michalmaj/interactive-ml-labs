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
SCREEN_MARGIN: Final[Size] = (120, 120)


def choose_adaptive_window_size(
    display_size: Size,
    *,
    presets: tuple[Size, ...] = WINDOWED_RESOLUTIONS,
    fallback: Size = DEFAULT_RESOLUTION,
    margin: Size = SCREEN_MARGIN,
) -> Size:
    """Choose the largest preset that fits within the current display."""
    display_width, display_height = display_size
    margin_width, margin_height = margin
    available_width = max(0, display_width - margin_width)
    available_height = max(0, display_height - margin_height)

    fitting_presets = [
        preset
        for preset in presets
        if preset[0] <= available_width and preset[1] <= available_height
    ]
    if not fitting_presets:
        return fallback

    return max(fitting_presets, key=lambda preset: preset[0] * preset[1])


def center_rect(container_size: Size, content_size: Size) -> RectTuple:
    """Return a centered rect tuple for fixed-size content."""
    container_width, container_height = container_size
    content_width, content_height = content_size

    left = max(0, (container_width - content_width) // 2)
    top = max(0, (container_height - content_height) // 2)

    return (left, top, content_width, content_height)


def scale_rect_to_fit(container_size: Size, content_size: Size) -> RectTuple:
    """Return a centered rect for content scaled to fit its container."""
    container_width, container_height = container_size
    content_width, content_height = content_size
    if content_width <= 0 or content_height <= 0:
        return (0, 0, 0, 0)

    scale = min(container_width / content_width, container_height / content_height)
    scaled_width = round(content_width * scale)
    scaled_height = round(content_height * scale)

    return center_rect(container_size, (scaled_width, scaled_height))
