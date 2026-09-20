"""Tests for the shell settings screen renderer."""

import pygame
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.screens.settings_screen import (
    SettingsScreenColors,
    SettingsScreenFonts,
    SettingsScreenRenderer,
)
from interactive_ml_labs.settings import AppSettings


def _fonts() -> SettingsScreenFonts:
    return SettingsScreenFonts(
        title=make_ui_font(44, bold=True),
        body=make_ui_font(22),
        small=make_ui_font(18),
    )


def _colors() -> SettingsScreenColors:
    return SettingsScreenColors(
        text=(235, 238, 241),
        muted_text=(166, 173, 181),
        panel=(38, 43, 49),
        selected_panel=(55, 97, 126),
        border=(72, 79, 88),
    )


def test_settings_screen_returns_menu_hitboxes(monkeypatch) -> None:
    """Settings renderer should return one selectable item for each option."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = SettingsScreenRenderer()

        menu_items = renderer.render(
            surface,
            settings=AppSettings(),
            selected_index=0,
            fonts=_fonts(),
            colors=_colors(),
            footer_y=670,
        )

        assert len(menu_items) == 9
        assert menu_items[0].label == "Language: English"
        assert menu_items[0].rect == pygame.Rect(80, 168, 760, 44)
        assert menu_items[-1].label == "Back"
    finally:
        pygame.quit()


def test_settings_screen_uses_polish_labels(monkeypatch) -> None:
    """Settings renderer should localize labels from app settings."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = SettingsScreenRenderer()

        menu_items = renderer.render(
            surface,
            settings=AppSettings(language="pl", fullscreen_enabled=True),
            selected_index=0,
            fonts=_fonts(),
            colors=_colors(),
            footer_y=670,
        )

        assert menu_items[0].label == "Język: Polski"
        assert menu_items[1].label == "Pełny ekran: Włączone"
        assert menu_items[-1].label == "Wróć"
    finally:
        pygame.quit()


def test_settings_screen_draws_selected_row(monkeypatch) -> None:
    """Selected settings row should use the selected panel color."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    try:
        surface = pygame.Surface((1280, 720))
        renderer = SettingsScreenRenderer()
        colors = _colors()

        menu_items = renderer.render(
            surface,
            settings=AppSettings(),
            selected_index=2,
            fonts=_fonts(),
            colors=colors,
            footer_y=670,
        )

        assert surface.get_at(menu_items[2].rect.center)[:3] == colors.selected_panel
        assert surface.get_at(menu_items[1].rect.center)[:3] == colors.panel
    finally:
        pygame.quit()
