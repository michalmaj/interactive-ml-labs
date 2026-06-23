"""Small shared drawing helpers for native Pygame lab scenes."""

from __future__ import annotations

from typing import Final

import pygame

DEFAULT_PANEL_RADIUS: Final[int] = 8
ELLIPSIS: Final[str] = "..."


def draw_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    *,
    border_radius: int = DEFAULT_PANEL_RADIUS,
) -> None:
    """Draw a rounded scene panel."""
    pygame.draw.rect(surface, color, rect, border_radius=border_radius)


def draw_text(
    surface: pygame.Surface,
    text: str,
    position: tuple[int, int],
    font: pygame.font.Font,
    color: tuple[int, int, int],
) -> None:
    """Draw one line of text."""
    surface.blit(font.render(text, True, color), position)


def draw_wrapped_text(
    surface: pygame.Surface,
    text: str,
    position: tuple[int, int],
    width: int,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    *,
    line_height: int,
    max_bottom: int | None = None,
) -> int:
    """Draw wrapped text and return the next y coordinate."""
    x, y = position
    lines = wrap_text(text, width, font)
    if max_bottom is not None:
        lines = _clip_lines_to_bottom(lines, y, line_height, max_bottom, width, font)

    for line in lines:
        draw_text(surface, line, (x, y), font, color)
        y += line_height

    return y


def wrap_text(text: str, width: int, font: pygame.font.Font) -> list[str]:
    """Wrap text into lines that fit the requested width."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def fit_with_ellipsis(text: str, width: int, font: pygame.font.Font) -> str:
    """Trim text so a clipped line still communicates that more text exists."""
    if font.size(text + ELLIPSIS)[0] <= width:
        return text + ELLIPSIS

    trimmed = text
    while trimmed and font.size(trimmed.rstrip() + ELLIPSIS)[0] > width:
        trimmed = trimmed[:-1]

    if not trimmed:
        return ELLIPSIS

    return trimmed.rstrip() + ELLIPSIS


def _clip_lines_to_bottom(
    lines: list[str],
    y: int,
    line_height: int,
    max_bottom: int,
    width: int,
    font: pygame.font.Font,
) -> list[str]:
    """Return only the wrapped lines that fit inside a bottom boundary."""
    available_lines = max(0, (max_bottom - y) // line_height)
    if available_lines == 0:
        return []
    if len(lines) <= available_lines:
        return lines
    return [
        *lines[: available_lines - 1],
        fit_with_ellipsis(lines[available_lines - 1], width, font),
    ]
