"""Tests for shared native lab UI drawing helpers."""

import pygame
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.ui_helpers import (
    draw_panel,
    draw_text,
    draw_wrapped_text,
    fit_with_ellipsis,
    wrap_text,
)


def test_wrap_text_keeps_lines_inside_width(monkeypatch) -> None:
    """Wrapped lines should fit the requested pixel width."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        font = make_ui_font(15)
        width = 120
        lines = wrap_text(
            "Validation score should guide model choice before test score is opened.",
            width,
            font,
        )

        assert len(lines) > 1
        assert all(font.size(line)[0] <= width for line in lines)
    finally:
        pygame.quit()


def test_draw_wrapped_text_returns_next_y_and_draws_inside_surface(monkeypatch) -> None:
    """Wrapped text drawing should return the next baseline and keep pixels in bounds."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((240, 160), pygame.SRCALPHA)
        font = make_ui_font(15)
        next_y = draw_wrapped_text(
            surface,
            "A short shared helper prevents every scene from copying wrapping code.",
            (12, 18),
            150,
            font,
            (240, 240, 240),
            line_height=18,
        )

        assert next_y > 18
        assert surface.get_bounding_rect().right <= surface.get_width()
        assert surface.get_bounding_rect().bottom <= surface.get_height()
    finally:
        pygame.quit()


def test_draw_wrapped_text_clips_to_bottom_with_ellipsis(monkeypatch) -> None:
    """Long clipped text should stay under max_bottom and show an ellipsis."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((260, 160), pygame.SRCALPHA)
        font = make_ui_font(15)
        next_y = draw_wrapped_text(
            surface,
            "This is intentionally long text that should be clipped to one or two lines.",
            (12, 18),
            120,
            font,
            (240, 240, 240),
            line_height=18,
            max_bottom=54,
        )

        assert next_y <= 54
        assert surface.get_bounding_rect().bottom <= 54
        assert fit_with_ellipsis("clipped text", 90, font).endswith("...")
    finally:
        pygame.quit()


def test_draw_panel_and_text_make_surface_non_empty(monkeypatch) -> None:
    """Panel and text helpers should draw visible pixels."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((220, 120), pygame.SRCALPHA)
        draw_panel(surface, pygame.Rect(10, 10, 160, 80), (34, 39, 45))
        draw_text(surface, "Shared UI", (24, 34), make_ui_font(18), (236, 239, 242))

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()
