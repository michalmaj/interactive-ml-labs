"""Font helpers for the unified Pygame shell."""

from __future__ import annotations

from typing import Final

import pygame

POLISH_FONT_CANDIDATES: Final[tuple[str, ...]] = (
    "arial",
    "dejavusans",
    "noto sans",
    "liberation sans",
    "segoe ui",
    "helvetica",
)

POLISH_SAMPLE_TEXT: Final[str] = "Zażółć gęślą jaźń"


def make_ui_font(size: int, *, bold: bool = False) -> pygame.font.Font:
    """Create a UI font with reasonable Unicode coverage for Polish text."""
    return pygame.font.SysFont(POLISH_FONT_CANDIDATES, size, bold=bold)
