"""Pure scrolling helpers for the unified app shell."""

from __future__ import annotations

import pygame


def clamp_scroll_offset(scroll_offset: int, max_scroll: int) -> int:
    """Clamp a scroll offset to the valid range."""
    return max(0, min(scroll_offset, max_scroll))


def content_max_scroll(content_end: int, scroll_offset: int, viewport_bottom: int) -> int:
    """Return max scroll for rendered content ending at a viewport position."""
    return max(0, content_end + scroll_offset - viewport_bottom)


def list_max_scroll(
    item_count: int,
    *,
    item_height: int,
    item_pitch: int,
    viewport_height: int,
) -> int:
    """Return max scroll for a vertical list with fixed item pitch."""
    if item_count <= 0:
        return 0

    content_height = ((item_count - 1) * item_pitch) + item_height
    return max(0, content_height - max(0, viewport_height))


def scroll_offset_for_selected_item(
    *,
    selected_index: int,
    scroll_offset: int,
    viewport_height: int,
    item_height: int,
    item_pitch: int,
    max_scroll: int,
) -> int:
    """Return scroll offset that keeps the selected fixed-pitch item visible."""
    selected_top = selected_index * item_pitch
    selected_bottom = selected_top + item_height
    viewport_height = max(0, viewport_height)

    if selected_top < scroll_offset:
        scroll_offset = selected_top
    elif selected_bottom > scroll_offset + viewport_height:
        scroll_offset = selected_bottom - viewport_height

    return clamp_scroll_offset(scroll_offset, max_scroll)


def selected_index_after_scroll(
    *,
    selected_index: int,
    scroll_offset: int,
    item_count: int,
    viewport_height: int,
    item_height: int,
    item_pitch: int,
) -> int:
    """Return a selected index that stays visible after list scrolling."""
    if item_count <= 0:
        return 0

    selected_top = selected_index * item_pitch
    selected_bottom = selected_top + item_height
    viewport_height = max(0, viewport_height)
    if scroll_offset <= selected_top and selected_bottom <= scroll_offset + viewport_height:
        return max(0, min(item_count - 1, selected_index))

    first_visible = scroll_offset // item_pitch
    return max(0, min(item_count - 1, first_visible))


def scrollbar_rects(
    *,
    x: int,
    top: int,
    bottom: int,
    scroll_offset: int,
    max_scroll: int,
    width: int,
    min_thumb_height: int,
) -> tuple[pygame.Rect, pygame.Rect]:
    """Return track and thumb rectangles for a vertical scrollbar."""
    track_height = max(0, bottom - top)
    track_rect = pygame.Rect(x, top, width, track_height)
    if max_scroll <= 0 or track_height <= 0:
        return track_rect, pygame.Rect(x, top, width, track_height)

    visible_ratio = track_height / (track_height + max_scroll)
    thumb_height = max(min_thumb_height, round(track_height * visible_ratio))
    thumb_height = min(track_height, thumb_height)
    thumb_range = max(1, track_height - thumb_height)
    offset = clamp_scroll_offset(scroll_offset, max_scroll)
    thumb_y = top + round(thumb_range * offset / max_scroll)
    thumb_rect = pygame.Rect(track_rect.x, thumb_y, track_rect.width, thumb_height)
    return track_rect, thumb_rect


def scroll_offset_from_thumb_y(
    *,
    thumb_y: int,
    top: int,
    bottom: int,
    thumb_height: int,
    max_scroll: int,
) -> int:
    """Return scroll offset represented by a visual scrollbar thumb y-position."""
    if max_scroll <= 0:
        return 0

    thumb_range = max(1, (bottom - top) - thumb_height)
    relative_thumb_y = clamp_scroll_offset(thumb_y - top, thumb_range)
    return round(relative_thumb_y / thumb_range * max_scroll)
