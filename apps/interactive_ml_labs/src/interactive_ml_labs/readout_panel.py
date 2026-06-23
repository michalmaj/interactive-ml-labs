"""Shared readout panel drawing helpers for native lab scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pygame

from interactive_ml_labs.ui_helpers import draw_panel, draw_text, draw_wrapped_text

ReadoutRow = tuple[str, str]
ReadoutOption = tuple[str, bool]

PADDING: Final[int] = 22
TITLE_OFFSET_Y: Final[int] = 20
CONTENT_OFFSET_Y: Final[int] = 68
ROW_GAP: Final[int] = 24
OPTION_GAP: Final[int] = 24
TAKEAWAY_BOTTOM_PADDING: Final[int] = 108
LINE_HEIGHT: Final[int] = 17


@dataclass(frozen=True, slots=True)
class ReadoutPanelFonts:
    """Fonts used by the shared readout panel."""

    heading: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class ReadoutPanelColors:
    """Color palette used by the shared readout panel."""

    panel: tuple[int, int, int]
    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    secondary: tuple[int, int, int]


def draw_readout_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    title: str,
    rows: tuple[ReadoutRow, ...],
    options: tuple[ReadoutOption, ...],
    takeaway: str,
    fonts: ReadoutPanelFonts,
    colors: ReadoutPanelColors,
) -> None:
    """Draw a compact side readout with rows, selectable options, and a takeaway."""
    draw_panel(surface, rect, colors.panel)
    draw_text(
        surface,
        title,
        (rect.x + PADDING, rect.y + TITLE_OFFSET_Y),
        fonts.heading,
        colors.text,
    )

    y = rect.y + CONTENT_OFFSET_Y
    row_width = rect.width - (PADDING * 2)
    for label, value in rows:
        y = _draw_readout_row(
            surface, label, value, (rect.x + PADDING, y), row_width, fonts, colors
        )
        y += 5

    y += 10
    options_bottom = rect.bottom - TAKEAWAY_BOTTOM_PADDING - 14
    for option_label, selected in options:
        if y > options_bottom:
            break
        color = colors.accent if selected else colors.muted_text
        draw_text(surface, option_label, (rect.x + PADDING, y), fonts.small, color)
        y += OPTION_GAP

    draw_wrapped_text(
        surface,
        takeaway,
        (rect.x + PADDING, rect.bottom - TAKEAWAY_BOTTOM_PADDING),
        rect.width - (PADDING * 2),
        fonts.small,
        colors.secondary,
        line_height=LINE_HEIGHT,
        max_bottom=rect.bottom - PADDING,
    )


def _draw_readout_row(
    surface: pygame.Surface,
    label: str,
    value: str,
    position: tuple[int, int],
    width: int,
    fonts: ReadoutPanelFonts,
    colors: ReadoutPanelColors,
) -> int:
    """Draw one row and return the next y coordinate."""
    x, y = position
    line = f"{label}: {value}"
    if fonts.small.size(line)[0] <= width:
        draw_text(surface, line, (x, y), fonts.small, colors.text)
        return y + ROW_GAP

    label_text = f"{label}:"
    draw_text(surface, label_text, (x, y), fonts.small, colors.muted_text)
    label_width = fonts.small.size(label_text)[0]
    if label_width + 126 > width:
        return (
            draw_wrapped_text(
                surface,
                value,
                (x, y + LINE_HEIGHT),
                width,
                fonts.small,
                colors.text,
                line_height=LINE_HEIGHT,
            )
            + 7
        )

    value_x = x + label_width + 6
    value_width = width - (value_x - x)
    return (
        draw_wrapped_text(
            surface,
            value,
            (value_x, y),
            value_width,
            fonts.small,
            colors.text,
            line_height=LINE_HEIGHT,
        )
        + 7
    )
