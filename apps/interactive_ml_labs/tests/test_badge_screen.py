"""Tests for the shell badge gallery renderer."""

import pygame
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.screens.badge_screen import (
    BadgeGalleryPath,
    BadgeGalleryRenderer,
    BadgeScreenColors,
    BadgeScreenFonts,
)
from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import BadgeItem


def _fonts() -> BadgeScreenFonts:
    return BadgeScreenFonts(
        title=make_ui_font(44, bold=True),
        heading=make_ui_font(30, bold=True),
        body=make_ui_font(22),
        small=make_ui_font(18),
    )


def _colors() -> BadgeScreenColors:
    return BadgeScreenColors(
        background=(22, 25, 29),
        text=(235, 238, 241),
        muted_text=(166, 173, 181),
        accent=(113, 204, 152),
        panel=(38, 43, 49),
        selected_panel=(55, 97, 126),
        border=(72, 79, 88),
        unlocked_fill=(226, 176, 83),
        unlocked_outline=(250, 218, 139),
        locked_fill=(61, 68, 76),
        locked_outline=(109, 118, 128),
    )


def test_badge_screen_returns_viewport_and_back_hitbox(monkeypatch) -> None:
    """Badge gallery renderer should return scroll viewport and Back menu hitbox."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = BadgeGalleryRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(resolution=(1280, 720)),
            paths=[
                BadgeGalleryPath(
                    title="Path One",
                    progress_label="Badges unlocked: 1/2",
                    badges=[
                        BadgeItem(label="First Badge", unlocked=True),
                        BadgeItem(label="Second Badge", unlocked=False),
                    ],
                ),
            ],
            summary_label="Badges unlocked: 1/2",
            scroll_offset=0,
            content_bottom=636,
            footer_y=670,
            fonts=_fonts(),
            colors=_colors(),
        )

        assert result.viewport.x == 80
        assert result.viewport.width == 1040
        assert result.content_end > result.viewport.y
        assert [item.label for item in result.menu_items] == ["Back"]
        assert result.menu_items[0].rect == pygame.Rect(80, 574, 260, 54)
    finally:
        pygame.quit()


def test_badge_screen_uses_polish_back_label(monkeypatch) -> None:
    """Badge gallery renderer should localize shell controls."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = BadgeGalleryRenderer()

        result = renderer.render(
            surface,
            settings=AppSettings(language="pl", resolution=(1280, 720)),
            paths=[],
            summary_label="Odznaki odblokowane: 0/0",
            scroll_offset=0,
            content_bottom=636,
            footer_y=670,
            fonts=_fonts(),
            colors=_colors(),
        )

        assert [item.label for item in result.menu_items] == ["Wróć"]
    finally:
        pygame.quit()


def test_badge_screen_draws_locked_and_unlocked_medallions(monkeypatch) -> None:
    """Unlocked and locked badge icons should use different fill colors."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = BadgeGalleryRenderer()
        colors = _colors()

        renderer.render(
            surface,
            settings=AppSettings(resolution=(1280, 720)),
            paths=[
                BadgeGalleryPath(
                    title="Path One",
                    progress_label="Badges unlocked: 1/2",
                    badges=[
                        BadgeItem(label="First Badge", unlocked=True),
                        BadgeItem(label="Second Badge", unlocked=False),
                    ],
                ),
            ],
            summary_label="Badges unlocked: 1/2",
            scroll_offset=0,
            content_bottom=636,
            footer_y=670,
            fonts=_fonts(),
            colors=colors,
        )

        assert surface.get_at((89, 325))[:3] == colors.unlocked_fill
        assert surface.get_at((89, 350))[:3] == colors.locked_fill
    finally:
        pygame.quit()
