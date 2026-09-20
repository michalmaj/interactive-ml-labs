"""Tests for unified shell scrolling helpers."""

import pygame
from interactive_ml_labs.shell_scrolling import (
    clamp_scroll_offset,
    content_max_scroll,
    list_max_scroll,
    scroll_offset_for_selected_item,
    scroll_offset_from_thumb_y,
    scrollbar_rects,
    selected_index_after_scroll,
)


def test_clamp_scroll_offset_keeps_offset_inside_range() -> None:
    """Scroll offsets should stay between zero and max scroll."""
    assert clamp_scroll_offset(-10, 100) == 0
    assert clamp_scroll_offset(42, 100) == 42
    assert clamp_scroll_offset(142, 100) == 100


def test_content_max_scroll_accounts_for_current_scroll_offset() -> None:
    """Rendered content bounds should convert to a stable max scroll offset."""
    assert content_max_scroll(content_end=500, scroll_offset=0, viewport_bottom=600) == 0
    assert content_max_scroll(content_end=760, scroll_offset=40, viewport_bottom=600) == 200


def test_list_max_scroll_handles_empty_and_overflowing_lists() -> None:
    """Fixed-pitch menu lists should report scroll only when content overflows."""
    assert list_max_scroll(0, item_height=54, item_pitch=70, viewport_height=300) == 0
    assert list_max_scroll(3, item_height=54, item_pitch=70, viewport_height=300) == 0
    assert list_max_scroll(8, item_height=54, item_pitch=70, viewport_height=300) == 244


def test_scroll_offset_for_selected_item_reveals_item_above_viewport() -> None:
    """Selection visibility should scroll upward when the item is above view."""
    assert (
        scroll_offset_for_selected_item(
            selected_index=2,
            scroll_offset=240,
            viewport_height=180,
            item_height=54,
            item_pitch=70,
            max_scroll=400,
        )
        == 140
    )


def test_scroll_offset_for_selected_item_reveals_item_below_viewport() -> None:
    """Selection visibility should scroll downward when the item is below view."""
    assert (
        scroll_offset_for_selected_item(
            selected_index=6,
            scroll_offset=100,
            viewport_height=180,
            item_height=54,
            item_pitch=70,
            max_scroll=400,
        )
        == 294
    )


def test_selected_index_after_scroll_keeps_visible_selection_when_possible() -> None:
    """Mouse-wheel scroll should keep selection when the selected item is visible."""
    assert (
        selected_index_after_scroll(
            selected_index=3,
            scroll_offset=140,
            item_count=8,
            viewport_height=180,
            item_height=54,
            item_pitch=70,
        )
        == 3
    )


def test_selected_index_after_scroll_moves_to_first_visible_item() -> None:
    """Mouse-wheel scroll should move selection to the first visible item if needed."""
    assert (
        selected_index_after_scroll(
            selected_index=0,
            scroll_offset=210,
            item_count=8,
            viewport_height=180,
            item_height=54,
            item_pitch=70,
        )
        == 3
    )


def test_scrollbar_rects_maps_scroll_offset_to_thumb_position() -> None:
    """Scrollbar thumb geometry should reflect the clamped scroll offset."""
    track, thumb = scrollbar_rects(
        x=20,
        top=100,
        bottom=300,
        scroll_offset=50,
        max_scroll=100,
        width=4,
        min_thumb_height=36,
    )

    assert track == pygame.Rect(20, 100, 4, 200)
    assert thumb.width == 4
    assert thumb.height == 133
    assert thumb.y == 134


def test_scrollbar_rects_handles_non_scrollable_content() -> None:
    """Non-scrollable content should return a full-height thumb."""
    track, thumb = scrollbar_rects(
        x=20,
        top=100,
        bottom=300,
        scroll_offset=0,
        max_scroll=0,
        width=4,
        min_thumb_height=36,
    )

    assert track == pygame.Rect(20, 100, 4, 200)
    assert thumb == pygame.Rect(20, 100, 4, 200)


def test_scroll_offset_from_thumb_y_clamps_track_positions() -> None:
    """Dragging outside the visual track should clamp to list bounds."""
    assert (
        scroll_offset_from_thumb_y(
            thumb_y=100,
            top=100,
            bottom=300,
            thumb_height=100,
            max_scroll=500,
        )
        == 0
    )
    assert (
        scroll_offset_from_thumb_y(
            thumb_y=250,
            top=100,
            bottom=300,
            thumb_height=100,
            max_scroll=500,
        )
        == 500
    )
