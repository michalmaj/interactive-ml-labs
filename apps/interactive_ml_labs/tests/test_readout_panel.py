"""Tests for shared native lab readout panel drawing."""

import pygame
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.readout_panel import (
    ReadoutPanelColors,
    ReadoutPanelFonts,
    draw_readout_panel,
)


def test_readout_panel_draws_compact_rows_and_takeaway(monkeypatch) -> None:
    """The shared readout panel should draw a non-empty compact panel."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((420, 480))
        draw_readout_panel(
            surface,
            pygame.Rect(0, 0, 420, 480),
            title="Model selection",
            rows=(
                ("preset", "Noisy validation"),
                ("train score", "98%"),
                ("validation score", "74%"),
                ("diagnosis", "overfit"),
            ),
            options=(("1. Noisy validation", True), ("2. Small dataset", False)),
            takeaway="Validation should guide the choice before test is opened.",
            fonts=ReadoutPanelFonts(
                heading=make_ui_font(23, bold=True),
                small=make_ui_font(15),
            ),
            colors=ReadoutPanelColors(
                panel=(34, 39, 45),
                text=(236, 239, 242),
                muted_text=(166, 174, 184),
                accent=(118, 205, 247),
                secondary=(248, 183, 96),
            ),
        )

        assert surface.get_bounding_rect().width > 0
        assert surface.get_bounding_rect().height > 0
    finally:
        pygame.quit()


def test_readout_panel_wraps_long_values_without_crashing(monkeypatch) -> None:
    """Long readout values should wrap instead of requiring scene-specific fixes."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        surface = pygame.Surface((300, 360))
        draw_readout_panel(
            surface,
            pygame.Rect(0, 0, 300, 360),
            title="Kontrola",
            rows=(
                (
                    "najdłuższy opis",
                    "bardzo długi tekst, który musi zmieścić się w panelu bez nakładania",
                ),
                ("diagnosis", "do sprawdzenia"),
            ),
            options=(("1. Opcja", True),),
            takeaway="Krótki wniosek na dole panelu też powinien zostać zawinięty.",
            fonts=ReadoutPanelFonts(
                heading=make_ui_font(23, bold=True),
                small=make_ui_font(15),
            ),
            colors=ReadoutPanelColors(
                panel=(34, 39, 45),
                text=(236, 239, 242),
                muted_text=(166, 174, 184),
                accent=(118, 205, 247),
                secondary=(248, 183, 96),
            ),
        )

        assert surface.get_bounding_rect().right <= surface.get_width()
        assert surface.get_bounding_rect().bottom <= surface.get_height()
    finally:
        pygame.quit()


def test_readout_panel_clips_long_takeaway_to_panel_bounds(monkeypatch) -> None:
    """Long takeaway text should stay inside the readout panel."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()

    try:
        panel_rect = pygame.Rect(0, 0, 300, 260)
        surface = pygame.Surface((340, 360), pygame.SRCALPHA)
        draw_readout_panel(
            surface,
            panel_rect,
            title="Odczyt",
            rows=(
                ("dataset", "długi opis"),
                ("diagnosis", "do sprawdzenia"),
            ),
            options=(("1. Opcja", True), ("2. Druga opcja", False)),
            takeaway=(
                "To jest bardzo długi wniosek, który w praktycznym labie mógłby "
                "mieć kilka linijek i nie powinien wychodzić poza dolną krawędź panelu."
            ),
            fonts=ReadoutPanelFonts(
                heading=make_ui_font(23, bold=True),
                small=make_ui_font(15),
            ),
            colors=ReadoutPanelColors(
                panel=(34, 39, 45),
                text=(236, 239, 242),
                muted_text=(166, 174, 184),
                accent=(118, 205, 247),
                secondary=(248, 183, 96),
            ),
        )

        assert surface.get_bounding_rect().bottom <= panel_rect.bottom
    finally:
        pygame.quit()
