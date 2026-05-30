"""Tests for unified app shell font helpers."""

from __future__ import annotations

import pygame
from interactive_ml_labs import POLISH_SAMPLE_TEXT, make_ui_font


def test_ui_font_renders_polish_sample_text(monkeypatch) -> None:
    """UI font helper should render Polish diacritics without crashing."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        font = make_ui_font(24)
        rendered = font.render(POLISH_SAMPLE_TEXT, True, (255, 255, 255))
    finally:
        pygame.quit()

    assert rendered.get_width() > 0
    assert rendered.get_height() > 0
