"""Minimal Pygame shell for Interactive ML Labs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pygame

from interactive_ml_labs.manifest import DemoManifest, LocalizedText
from interactive_ml_labs.registry import LEVEL_NAMES, demos_for_level, levels_from_manifests
from interactive_ml_labs.settings import AppContext, AppSettings

FPS: Final[int] = 60
BACKGROUND: Final[tuple[int, int, int]] = (22, 25, 29)
PANEL: Final[tuple[int, int, int]] = (38, 43, 49)
PANEL_SELECTED: Final[tuple[int, int, int]] = (55, 97, 126)
TEXT: Final[tuple[int, int, int]] = (235, 238, 241)
MUTED_TEXT: Final[tuple[int, int, int]] = (166, 173, 181)
ACCENT: Final[tuple[int, int, int]] = (113, 204, 152)


class ScreenName(StrEnum):
    """Top-level shell screens."""

    LANGUAGE = "language"
    LEVELS = "levels"
    DEMOS = "demos"
    INTRO = "intro"
    PLACEHOLDER_DEMO = "placeholder_demo"
    PAUSE = "pause"


@dataclass(slots=True)
class MenuItem:
    """Clickable/selectable menu item."""

    label: str
    rect: pygame.Rect


class UnifiedAppShell:
    """Small first slice of the unified Pygame app shell."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the shell."""
        pygame.init()

        self.context = AppContext(settings=settings or AppSettings())
        self.screen = pygame.display.set_mode(self.context.settings.resolution)
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("arial", 44, bold=True)
        self.font_heading = pygame.font.SysFont("arial", 30, bold=True)
        self.font_body = pygame.font.SysFont("arial", 22)
        self.font_small = pygame.font.SysFont("arial", 18)
        self.running = True
        self.screen_name = ScreenName.LANGUAGE
        self.previous_screen = ScreenName.INTRO
        self.selected_index = 0
        self.selected_demo: DemoManifest | None = None
        self.menu_items: list[MenuItem] = []
        self.help_visible = False

        pygame.display.set_caption("Interactive ML Labs")

    def run(self) -> None:
        """Run the shell event loop."""
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                self._handle_events()
                self._update(dt)
                self._render()
        finally:
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

    def _handle_keydown(self, key: int) -> None:
        handlers = {
            pygame.K_UP: self._move_up,
            pygame.K_DOWN: self._move_down,
            pygame.K_RETURN: self._activate_selected,
            pygame.K_SPACE: self._activate_selected,
            pygame.K_ESCAPE: self._escape,
            pygame.K_h: self._toggle_help,
        }
        handler = handlers.get(key)

        if handler is not None:
            handler()

    def _handle_mouse_click(self, position: tuple[int, int]) -> None:
        for index, item in enumerate(self.menu_items):
            if item.rect.collidepoint(position):
                self.selected_index = index
                self._activate_selected()
                return

    def _update(self, dt: float) -> None:
        _ = dt

    def _render(self) -> None:
        self.screen.fill(BACKGROUND)

        renderers = {
            ScreenName.LANGUAGE: self._render_language,
            ScreenName.LEVELS: self._render_levels,
            ScreenName.DEMOS: self._render_demos,
            ScreenName.INTRO: self._render_intro,
            ScreenName.PLACEHOLDER_DEMO: self._render_placeholder_demo,
            ScreenName.PAUSE: self._render_pause,
        }
        renderers[self.screen_name]()

        if self.help_visible and self.screen_name != ScreenName.PAUSE:
            self._render_help_overlay()

        pygame.display.flip()

    def _render_language(self) -> None:
        self._draw_title("Interactive ML Labs", "Choose language / Wybierz jezyk")
        self._draw_menu(["English", "Polski"], top=230)
        self._draw_footer("Enter: select | Up/Down: move | Mouse: select")

    def _render_levels(self) -> None:
        language = self.context.settings.language
        labels = [LEVEL_NAMES[level].for_language(language) for level in levels_from_manifests()]
        self._draw_title("Interactive ML Labs", self._text("Select level", "Wybierz poziom"))
        self._draw_menu(labels, top=220)
        self._draw_footer(self._text("Esc: language | Enter: demos", "Esc: jezyk | Enter: dema"))

    def _render_demos(self) -> None:
        demos = self._current_level_demos()
        labels = [demo.title.for_language(self.context.settings.language) for demo in demos]
        level_name = LEVEL_NAMES[self.context.current_level or 1].for_language(
            self.context.settings.language,
        )

        self._draw_title(level_name, self._text("Select demo", "Wybierz demo"))
        self._draw_menu(labels, top=190)
        self._draw_footer(self._text("Esc: levels | Enter: intro", "Esc: poziomy | Enter: intro"))

    def _render_intro(self) -> None:
        demo = self._require_demo()
        language = self.context.settings.language
        y = 120

        self._draw_text(demo.title.for_language(language), (80, y), self.font_title, TEXT)
        y += 70
        self._draw_wrapped(
            demo.summary.for_language(language),
            (80, y),
            850,
            self.font_body,
            MUTED_TEXT,
        )
        y += 80
        self._draw_text(self._text("Objectives", "Cele"), (80, y), self.font_heading, ACCENT)
        y += 42

        for objective in demo.objectives:
            self._draw_wrapped(
                "- " + objective.for_language(language),
                (100, y),
                900,
                self.font_body,
                TEXT,
            )
            y += 38

        y += 20
        self._draw_text(self._text("Controls", "Sterowanie"), (80, y), self.font_heading, ACCENT)
        y += 40

        for control in demo.controls:
            label = f"{control.key}: {control.action.for_language(language)}"
            self._draw_text(label, (100, y), self.font_body, TEXT)
            y += 30

        self._draw_footer(self._text("Enter: start | Esc: demos", "Enter: start | Esc: dema"))

    def _render_placeholder_demo(self) -> None:
        demo = self._require_demo()
        language = self.context.settings.language
        self._draw_title(
            demo.title.for_language(language),
            self._text("Demo integration placeholder", "Placeholder integracji demo"),
        )
        self._draw_wrapped(
            self._text(
                "This shell is ready. The real demo scene will be connected in a later slice.",
                "Shell jest gotowy. Prawdziwa scena demo zostanie podpieta w kolejnym kroku.",
            ),
            (120, 280),
            900,
            self.font_body,
            TEXT,
        )
        self._draw_footer(self._text("Esc: pause | H: help", "Esc: pauza | H: pomoc"))

    def _render_pause(self) -> None:
        self._draw_title(self._text("Paused", "Pauza"), self._text("Shell menu", "Menu aplikacji"))
        labels = [
            self._text("Resume", "Wroc"),
            self._text("Help", "Pomoc"),
            self._text("Back to demos", "Wroc do dem"),
            self._text("Quit", "Zamknij"),
        ]
        self._draw_menu(labels, top=220)
        self._draw_footer("Esc: resume | Enter: select")

    def _draw_title(self, title: str, subtitle: str) -> None:
        self._draw_text(title, (80, 70), self.font_title, TEXT)
        self._draw_text(subtitle, (82, 128), self.font_body, MUTED_TEXT)

    def _draw_menu(self, labels: list[str], *, top: int) -> None:
        self.menu_items = []
        width = 760
        height = 54
        left = 80

        for index, label in enumerate(labels):
            rect = pygame.Rect(left, top + index * 70, width, height)
            color = PANEL_SELECTED if index == self.selected_index else PANEL
            pygame.draw.rect(self.screen, color, rect, border_radius=8)
            pygame.draw.rect(self.screen, (72, 79, 88), rect, width=1, border_radius=8)
            self._draw_text(label, (rect.x + 20, rect.y + 14), self.font_body, TEXT)
            self.menu_items.append(MenuItem(label=label, rect=rect))

    def _draw_footer(self, text: str) -> None:
        _, height = self.context.settings.resolution
        self._draw_text(text, (80, height - 50), self.font_small, MUTED_TEXT)

    def _draw_text(
        self,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        rendered = font.render(text, antialias=True, color=color)
        self.screen.blit(rendered, position)

    def _draw_wrapped(
        self,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        words = text.split()
        line = ""
        x, y = position

        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width:
                line = candidate
                continue

            self._draw_text(line, (x, y), font, color)
            y += font.get_linesize()
            line = word

        if line:
            self._draw_text(line, (x, y), font, color)

    def _render_help_overlay(self) -> None:
        width, height = self.context.settings.resolution
        rect = pygame.Rect(120, 130, width - 240, height - 260)
        pygame.draw.rect(self.screen, (12, 14, 17), rect, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, rect, width=2, border_radius=8)
        self._draw_text(
            self._text("Help", "Pomoc"),
            (rect.x + 32, rect.y + 28),
            self.font_heading,
            TEXT,
        )
        self._draw_wrapped(
            self._text(
                "Use arrow keys, Enter, Esc, H, or the mouse. "
                "Demo-specific help will come from manifests.",
                "Uzywaj strzalek, Enter, Esc, H albo myszy. "
                "Pomoc dla dem bedzie pochodzic z manifestow.",
            ),
            (rect.x + 32, rect.y + 82),
            rect.width - 64,
            self.font_body,
            MUTED_TEXT,
        )

    def _move_up(self) -> None:
        self.selected_index = max(0, self.selected_index - 1)

    def _move_down(self) -> None:
        item_count = self._current_menu_item_count()
        self.selected_index = min(item_count - 1, self.selected_index + 1)

    def _activate_selected(self) -> None:
        actions = {
            ScreenName.LANGUAGE: self._select_language,
            ScreenName.LEVELS: self._select_level,
            ScreenName.DEMOS: self._select_demo,
            ScreenName.INTRO: self._start_demo,
            ScreenName.PLACEHOLDER_DEMO: self._open_pause,
            ScreenName.PAUSE: self._select_pause_item,
        }
        actions[self.screen_name]()

    def _select_language(self) -> None:
        self.context.settings.language = "en" if self.selected_index == 0 else "pl"
        self._go_to(ScreenName.LEVELS)

    def _select_level(self) -> None:
        levels = levels_from_manifests()
        self.context.current_level = levels[self.selected_index]
        self._go_to(ScreenName.DEMOS)

    def _select_demo(self) -> None:
        demos = self._current_level_demos()
        self.selected_demo = demos[self.selected_index]
        self.context.selected_demo_id = self.selected_demo.id
        self._go_to(ScreenName.INTRO)

    def _start_demo(self) -> None:
        self._go_to(ScreenName.PLACEHOLDER_DEMO)

    def _select_pause_item(self) -> None:
        if self.selected_index == 0:
            self._resume()
        elif self.selected_index == 1:
            self.help_visible = not self.help_visible
        elif self.selected_index == 2:
            self.help_visible = False
            self._go_to(ScreenName.DEMOS)
        else:
            self.running = False

    def _escape(self) -> None:
        if self.screen_name == ScreenName.LANGUAGE:
            self.running = False
        elif self.screen_name == ScreenName.LEVELS:
            self._go_to(ScreenName.LANGUAGE)
        elif self.screen_name == ScreenName.DEMOS:
            self._go_to(ScreenName.LEVELS)
        elif self.screen_name == ScreenName.INTRO:
            self._go_to(ScreenName.DEMOS)
        elif self.screen_name == ScreenName.PLACEHOLDER_DEMO:
            self._open_pause()
        elif self.screen_name == ScreenName.PAUSE:
            self._resume()

    def _open_pause(self) -> None:
        self.previous_screen = self.screen_name
        self._go_to(ScreenName.PAUSE)

    def _resume(self) -> None:
        self._go_to(self.previous_screen)

    def _toggle_help(self) -> None:
        self.help_visible = not self.help_visible

    def _go_to(self, screen_name: ScreenName) -> None:
        self.screen_name = screen_name
        self.selected_index = 0
        self.menu_items = []

    def _current_menu_item_count(self) -> int:
        counts = {
            ScreenName.LANGUAGE: 2,
            ScreenName.LEVELS: len(levels_from_manifests()),
            ScreenName.DEMOS: len(self._current_level_demos()),
            ScreenName.INTRO: 1,
            ScreenName.PLACEHOLDER_DEMO: 1,
            ScreenName.PAUSE: 4,
        }
        return max(1, counts[self.screen_name])

    def _current_level_demos(self) -> tuple[DemoManifest, ...]:
        return demos_for_level(self.context.current_level or 1)

    def _require_demo(self) -> DemoManifest:
        if self.selected_demo is None:
            return self._current_level_demos()[0]

        return self.selected_demo

    def _text(self, en: str, pl: str) -> str:
        return LocalizedText(en=en, pl=pl).for_language(self.context.settings.language)


def main() -> None:
    """Run the unified app shell."""
    UnifiedAppShell().run()


if __name__ == "__main__":
    main()
