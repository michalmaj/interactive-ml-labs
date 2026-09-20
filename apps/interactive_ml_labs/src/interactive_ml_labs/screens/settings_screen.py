"""Renderer for the unified shell settings screen."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import MenuItem


@dataclass(frozen=True, slots=True)
class SettingsScreenFonts:
    """Fonts used by the settings screen renderer."""

    title: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class SettingsScreenColors:
    """Resolved colors used by the settings screen renderer."""

    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    panel: tuple[int, int, int]
    selected_panel: tuple[int, int, int]
    border: tuple[int, int, int]


class SettingsScreenRenderer:
    """Draw the shell settings screen and return clickable menu items."""

    def __init__(
        self,
        *,
        menu_left: int = 80,
        menu_top: int = 168,
        menu_width: int = 760,
        item_height: int = 44,
        item_pitch: int = 52,
    ) -> None:
        """Initialize static settings screen geometry."""
        self.menu_left = menu_left
        self.menu_top = menu_top
        self.menu_width = menu_width
        self.item_height = item_height
        self.item_pitch = item_pitch

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        selected_index: int,
        fonts: SettingsScreenFonts,
        colors: SettingsScreenColors,
        footer_y: int,
    ) -> list[MenuItem]:
        """Render the settings screen and return its menu items."""
        self._draw_text(
            surface,
            self._text(settings, "Settings", "Ustawienia"),
            (80, 70),
            fonts.title,
            colors.text,
        )
        self._draw_text(
            surface,
            self._text(settings, "In-memory app options", "Opcje aplikacji w tej sesji"),
            (82, 128),
            fonts.body,
            colors.muted_text,
        )
        menu_items = self._draw_menu(
            surface,
            self._labels(settings),
            selected_index=selected_index,
            fonts=fonts,
            colors=colors,
        )
        self._draw_text(
            surface,
            self._text(
                settings,
                "Enter: toggle/select | Esc/Backspace: back | Most comfort settings apply now",
                "Enter: przełącz | Esc/Backspace: wróć | Większość opcji komfortu działa od razu",
            ),
            (80, footer_y),
            fonts.small,
            colors.muted_text,
        )
        return menu_items

    def _labels(self, settings: AppSettings) -> list[str]:
        """Return localized settings menu labels."""
        return [
            self._text(settings, "Language: ", "Język: ") + self._language_label(settings),
            self._text(settings, "Fullscreen: ", "Pełny ekran: ")
            + self._on_off(settings, settings.fullscreen_enabled),
            self._text(settings, "Adaptive window size: ", "Adaptacyjny rozmiar okna: ")
            + self._on_off(settings, settings.adaptive_window_enabled),
            self._text(settings, "Fixed-scene scaling: ", "Skalowanie stałych scen: ")
            + self._on_off(settings, settings.fixed_scene_scaling_enabled),
            self._text(settings, "Large text: ", "Większy tekst: ")
            + self._on_off(settings, settings.large_text_enabled),
            self._text(settings, "High contrast: ", "Wysoki kontrast: ")
            + self._on_off(settings, settings.high_contrast_enabled),
            self._text(settings, "Colorblind-friendly palette: ", "Paleta przyjazna daltonizmowi: ")
            + self._on_off(settings, settings.colorblind_palette_enabled),
            self._text(settings, "Sound: ", "Dźwięk: ")
            + self._on_off(settings, settings.sound_enabled),
            self._text(settings, "Back", "Wróć"),
        ]

    def _draw_menu(
        self,
        surface: pygame.Surface,
        labels: list[str],
        *,
        selected_index: int,
        fonts: SettingsScreenFonts,
        colors: SettingsScreenColors,
    ) -> list[MenuItem]:
        """Draw settings menu rows and return their hitboxes."""
        menu_items: list[MenuItem] = []
        for index, label in enumerate(labels):
            rect = pygame.Rect(
                self.menu_left,
                self.menu_top + index * self.item_pitch,
                self.menu_width,
                self.item_height,
            )
            color = colors.selected_panel if index == selected_index else colors.panel
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)
            label_y = rect.y + max(0, (rect.height - fonts.body.get_height()) // 2)
            self._draw_text(surface, label, (rect.x + 20, label_y), fonts.body, colors.text)
            menu_items.append(MenuItem(label=label, rect=rect))

        return menu_items

    @staticmethod
    def _draw_text(
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one text run on the target surface."""
        surface.blit(font.render(text, True, color), position)

    @staticmethod
    def _text(settings: AppSettings, en: str, pl: str) -> str:
        """Return localized text for current app language."""
        return pl if settings.language == "pl" else en

    @staticmethod
    def _on_off(settings: AppSettings, value: bool) -> str:
        """Return a localized on/off label."""
        if settings.language == "pl":
            return "włączone" if value else "wyłączone"

        return "on" if value else "off"

    @staticmethod
    def _language_label(settings: AppSettings) -> str:
        """Return localized language label."""
        if settings.language == "pl":
            return "polski"

        return "English"
