"""Minimal Pygame shell for Interactive ML Labs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pygame

from interactive_ml_labs.display import Size, choose_adaptive_window_size, scale_rect_to_fit
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.manifest import DemoManifest, LocalizedText
from interactive_ml_labs.placeholder_scene import PlaceholderDemoScene
from interactive_ml_labs.registry import LEVEL_NAMES, demos_for_level, levels_from_manifests
from interactive_ml_labs.scene import (
    FixedSizeScene,
    Scene,
    SceneCommand,
    SceneCommandKind,
    SceneManager,
)
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
    DEMO = "demo"
    SETTINGS = "settings"
    PAUSE = "pause"


@dataclass(slots=True)
class MenuItem:
    """Clickable/selectable menu item."""

    label: str
    rect: pygame.Rect
    enabled: bool = True


class UnifiedAppShell:
    """Small first slice of the unified Pygame app shell."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        """Initialize the shell."""
        pygame.init()

        self.context = AppContext(settings=settings or AppSettings())
        self._apply_adaptive_window_size()
        self.screen = pygame.display.set_mode(
            self.context.settings.resolution,
            self._display_flags(),
        )
        self.clock = pygame.time.Clock()
        self.font_title = make_ui_font(44, bold=True)
        self.font_heading = make_ui_font(30, bold=True)
        self.font_body = make_ui_font(22)
        self.font_small = make_ui_font(18)
        self.running = True
        self.screen_name = ScreenName.LANGUAGE
        self.previous_screen = ScreenName.INTRO
        self.settings_return_screen = ScreenName.LEVELS
        self.selected_index = 0
        self.selected_demo: DemoManifest | None = None
        self.scene_manager = SceneManager()
        self.menu_items: list[MenuItem] = []
        self.help_visible = False
        self.mouse_position: tuple[int, int] = (0, 0)

        pygame.display.set_caption("Interactive ML Labs")

    def _apply_adaptive_window_size(self) -> None:
        """Apply opt-in adaptive window sizing before creating the display."""
        settings = self.context.settings
        if not settings.adaptive_window_enabled or settings.fullscreen_enabled:
            return

        display_info = pygame.display.Info()
        settings.resolution = choose_adaptive_window_size(
            (display_info.current_w, display_info.current_h),
        )

    def _display_flags(self) -> int:
        """Return Pygame display flags for current settings."""
        if self.context.settings.fullscreen_enabled:
            return pygame.FULLSCREEN

        return 0

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
            elif self.screen_name == ScreenName.DEMO and self._is_demo_event(event):
                self._handle_active_demo_event(event)
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

    def _is_demo_event(self, event: pygame.event.Event) -> bool:
        return not (event.type == pygame.KEYDOWN and event.key in {pygame.K_ESCAPE, pygame.K_h})

    def _handle_active_demo_event(self, event: pygame.event.Event) -> None:
        scene = self.scene_manager.current
        if scene is None:
            return

        scene_event = self._event_in_scene_coordinates(event, scene)
        if scene_event is None:
            return

        self._handle_scene_command(scene.handle_event(scene_event))

    def _event_in_scene_coordinates(
        self,
        event: pygame.event.Event,
        scene: Scene,
    ) -> pygame.event.Event | None:
        """Translate mouse positions from window space into fixed scene space."""
        fixed_scene_size = self._fixed_scene_size_for(scene)
        if fixed_scene_size is None or not hasattr(event, "pos"):
            return event

        target_rect = pygame.Rect(
            scale_rect_to_fit(self.context.settings.resolution, fixed_scene_size),
        )
        position = event.pos
        if not target_rect.collidepoint(position):
            return None

        logical_position = (
            round((position[0] - target_rect.x) * fixed_scene_size[0] / target_rect.width),
            round((position[1] - target_rect.y) * fixed_scene_size[1] / target_rect.height),
        )
        attributes = dict(event.dict)
        attributes["pos"] = logical_position
        return pygame.event.Event(event.type, attributes)

    def _handle_keydown(self, key: int) -> None:
        handlers = {
            pygame.K_UP: self._move_up,
            pygame.K_DOWN: self._move_down,
            pygame.K_RETURN: self._activate_selected,
            pygame.K_SPACE: self._activate_selected,
            pygame.K_ESCAPE: self._escape,
            pygame.K_BACKSPACE: self._escape,
            pygame.K_h: self._toggle_help,
            pygame.K_l: self._toggle_language,
            pygame.K_s: self._open_settings,
        }
        handler = handlers.get(key)

        if handler is not None:
            handler()

    def _handle_mouse_click(self, position: tuple[int, int]) -> None:
        if self._select_menu_item_at(position):
            self._activate_selected()

    def _handle_mouse_motion(self, position: tuple[int, int]) -> None:
        self.mouse_position = position
        self._select_menu_item_at(position)

    def _select_menu_item_at(self, position: tuple[int, int]) -> bool:
        for index, item in enumerate(self.menu_items):
            if item.enabled and item.rect.collidepoint(position):
                self.selected_index = index
                return True

        return False

    def _update(self, dt: float) -> None:
        if self.screen_name != ScreenName.DEMO:
            return

        scene = self.scene_manager.current
        if scene is None:
            return

        self._handle_scene_command(scene.update(dt))

    def _render(self) -> None:
        self.screen.fill(BACKGROUND)

        renderers = {
            ScreenName.LANGUAGE: self._render_language,
            ScreenName.LEVELS: self._render_levels,
            ScreenName.DEMOS: self._render_demos,
            ScreenName.INTRO: self._render_intro,
            ScreenName.DEMO: self._render_demo,
            ScreenName.SETTINGS: self._render_settings,
            ScreenName.PAUSE: self._render_pause,
        }
        renderers[self.screen_name]()

        if self.help_visible and self.screen_name != ScreenName.LANGUAGE:
            self._render_help_overlay()

        pygame.display.flip()

    def _render_language(self) -> None:
        self._draw_title("Interactive ML Labs", "Choose language / Wybierz język")
        self._draw_menu(["English", "Polski"], top=230)
        self._draw_footer("Enter: select | Up/Down: move | Mouse: select | S: settings | Esc: quit")

    def _render_levels(self) -> None:
        language = self.context.settings.language
        labels = [LEVEL_NAMES[level].for_language(language) for level in levels_from_manifests()]
        self._draw_title("Interactive ML Labs", self._text("Select level", "Wybierz poziom"))
        self._draw_menu(labels, top=220)
        self._draw_footer(
            self._text(
                "Enter: demos | Esc/Backspace: language | L: language",
                "Enter: dema | Esc/Backspace: wybór języka | S: ustawienia | L: zmień język",
            ),
        )

    def _render_demos(self) -> None:
        demos = self._current_level_demos()
        labels = [demo.title.for_language(self.context.settings.language) for demo in demos]
        level_name = LEVEL_NAMES[self.context.current_level or 1].for_language(
            self.context.settings.language,
        )

        self._draw_title(level_name, self._text("Select demo", "Wybierz demo"))
        self._draw_menu(labels, top=190)
        self._draw_footer(
            self._text(
                "Enter: intro | Esc/Backspace: levels | L: language",
                "Enter: intro | Esc/Backspace: poziomy | S: ustawienia | L: zmień język",
            ),
        )

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

        self._draw_footer(
            self._text(
                "Enter: start | Esc/Backspace: demos | L: language",
                "Enter: start | Esc/Backspace: lista dem | S: ustawienia | L: zmień język",
            ),
        )

    def _render_demo(self) -> None:
        scene = self.scene_manager.current
        if scene is not None and not isinstance(scene, PlaceholderDemoScene):
            self._render_scene(scene)
            return

        demo = self._require_demo()
        language = self.context.settings.language
        self._draw_title(
            demo.title.for_language(language),
            self._text("Demo integration placeholder", "Tu wkrótce pojawi się demo"),
        )
        self._draw_wrapped(
            self._text(
                "This shell is ready. The real demo scene will be connected in a later slice.",
                "Shell już działa. W kolejnym kroku podepniemy tutaj właściwą scenę demo.",
            ),
            (120, 280),
            900,
            self.font_body,
            TEXT,
        )
        if scene is not None:
            self._render_scene(scene)
        self._draw_footer(
            self._text(
                "Esc: pause | H: help | L: language",
                "Esc: pauza | H: pomoc | L: zmień język",
            ),
        )

    def _render_scene(self, scene: Scene) -> None:
        """Render a scene directly or through an opt-in fixed-size scaling buffer."""
        fixed_scene_size = self._fixed_scene_size_for(scene)
        if fixed_scene_size is None:
            scene.render(self.screen)
            return

        scene_surface = pygame.Surface(fixed_scene_size)
        scene.render(scene_surface)
        target_rect = scale_rect_to_fit(self.context.settings.resolution, fixed_scene_size)
        scaled_surface = pygame.transform.smoothscale(
            scene_surface,
            (target_rect[2], target_rect[3]),
        )
        self.screen.blit(scaled_surface, (target_rect[0], target_rect[1]))

    def _fixed_scene_size_for(self, scene: Scene) -> Size | None:
        """Return the scene's fixed logical size when scaling is enabled."""
        if not self.context.settings.fixed_scene_scaling_enabled:
            return None

        if not isinstance(scene, FixedSizeScene) or not self._is_valid_size(
            scene.fixed_scene_size,
        ):
            return None

        return scene.fixed_scene_size

    def _is_valid_size(self, value: object) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], int)
            and isinstance(value[1], int)
            and value[0] > 0
            and value[1] > 0
        )

    def _render_pause(self) -> None:
        self._draw_title(self._text("Paused", "Pauza"), self._text("Shell menu", "Menu aplikacji"))
        labels = [
            self._text("Resume", "Wróć"),
            self._text("Help", "Pomoc"),
            self._text("Restart", "Restart"),
            self._text("Settings", "Ustawienia"),
            self._text("Back to demos", "Lista dem"),
            self._text("Quit", "Zamknij"),
        ]
        self._draw_menu(labels, top=220)
        self._draw_footer(
            self._text(
                "Esc: resume | Enter: select | L: language",
                "Esc: wróć | Enter: wybierz | L: zmień język",
            ),
        )

    def _render_settings(self) -> None:
        settings = self.context.settings
        self._draw_title(
            self._text("Settings", "Ustawienia"),
            self._text("In-memory app options", "Opcje aplikacji w tej sesji"),
        )
        labels = [
            self._text("Language: ", "Język: ") + self._language_label(),
            self._text("Adaptive window size: ", "Adaptacyjny rozmiar okna: ")
            + self._on_off(settings.adaptive_window_enabled),
            self._text("Fixed-scene scaling: ", "Skalowanie stałych scen: ")
            + self._on_off(settings.fixed_scene_scaling_enabled),
            self._text("Sound: ", "Dźwięk: ") + self._on_off(settings.sound_enabled),
            self._text("Back", "Wróć"),
        ]
        self._draw_menu(labels, top=190)
        self._draw_footer(
            self._text(
                "Enter: toggle/select | Esc/Backspace: back | Adaptive size applies next launch",
                "Enter: przełącz | Esc/Backspace: wróć | Rozmiar okna od następnego startu",
            ),
        )

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
        rendered = font.render(text, True, color)
        self.screen.blit(rendered, position)

    def _draw_wrapped(
        self,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
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
            y += font.get_linesize()

        return y

    def _render_help_overlay(self) -> None:
        width, height = self.context.settings.resolution
        margin_x = min(120, max(32, width // 12))
        margin_y = min(90, max(32, height // 10))
        rect = pygame.Rect(
            margin_x,
            margin_y,
            width - 2 * margin_x,
            height - 2 * margin_y,
        )
        pygame.draw.rect(self.screen, (12, 14, 17), rect, border_radius=8)
        pygame.draw.rect(self.screen, ACCENT, rect, width=2, border_radius=8)
        language = self.context.settings.language
        demo = self._active_help_demo()

        self._draw_text(
            self._help_title(demo, language),
            (rect.x + 32, rect.y + 28),
            self.font_heading,
            TEXT,
        )

        y = rect.y + 78
        content_width = rect.width - 64
        if demo is None:
            self._draw_wrapped(
                self._text(
                    "Use arrow keys, Enter, Esc, H, or the mouse.",
                    "Używaj strzałek, Enter, Esc, H albo myszy.",
                ),
                (rect.x + 32, y),
                content_width,
                self.font_body,
                MUTED_TEXT,
            )
            return

        y = self._draw_wrapped(
            demo.summary.for_language(language),
            (rect.x + 32, y),
            content_width,
            self.font_body,
            MUTED_TEXT,
        )
        y += 18
        y = self._draw_help_section(
            self._text("Objectives", "Cele"),
            [objective.for_language(language) for objective in demo.objectives],
            rect.x + 32,
            y,
            content_width,
        )
        y += 12
        self._draw_help_section(
            self._text("Controls", "Sterowanie"),
            [
                f"{control.key}: {control.action.for_language(language)}"
                for control in demo.controls
            ],
            rect.x + 32,
            y,
            content_width,
        )

    def _draw_help_section(
        self,
        title: str,
        items: list[str],
        x: int,
        y: int,
        width: int,
    ) -> int:
        self._draw_text(title, (x, y), self.font_small, ACCENT)
        y += 28
        for item in items:
            y = self._draw_wrapped(f"- {item}", (x + 16, y), width - 16, self.font_small, TEXT)
            y += 4

        return y

    def _help_title(self, demo: DemoManifest | None, language: str) -> str:
        if demo is None:
            return self._text("Help", "Pomoc")

        return self._text("Help", "Pomoc") + " - " + demo.title.for_language(language)

    def _active_help_demo(self) -> DemoManifest | None:
        if self.screen_name in {ScreenName.INTRO, ScreenName.DEMO, ScreenName.PAUSE}:
            return self._require_demo()

        return None

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
            ScreenName.DEMO: self._open_pause,
            ScreenName.SETTINGS: self._select_settings_item,
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
        demo = self._require_demo()
        if demo.create_scene is not None:
            self.scene_manager.replace(demo.create_scene(self.context))
        self._go_to(ScreenName.DEMO)

    def _select_pause_item(self) -> None:
        if self.selected_index == 0:
            self._resume()
        elif self.selected_index == 1:
            self.help_visible = not self.help_visible
        elif self.selected_index == 2:
            self.help_visible = False
            self._restart_current_demo()
            self._go_to(ScreenName.DEMO)
        elif self.selected_index == 3:
            self._open_settings()
        elif self.selected_index == 4:
            self.help_visible = False
            self.scene_manager.clear()
            self._go_to(ScreenName.DEMOS)
        else:
            self.running = False

    def _select_settings_item(self) -> None:
        settings = self.context.settings
        if self.selected_index == 0:
            self._toggle_language()
        elif self.selected_index == 1:
            settings.adaptive_window_enabled = not settings.adaptive_window_enabled
        elif self.selected_index == 2:
            settings.fixed_scene_scaling_enabled = not settings.fixed_scene_scaling_enabled
        elif self.selected_index == 3:
            settings.sound_enabled = not settings.sound_enabled
        else:
            self._go_to(self.settings_return_screen)

    def _escape(self) -> None:
        if self.screen_name == ScreenName.LANGUAGE:
            self.running = False
        elif self.screen_name == ScreenName.LEVELS:
            self._go_to(ScreenName.LANGUAGE)
        elif self.screen_name == ScreenName.DEMOS:
            self._go_to(ScreenName.LEVELS)
        elif self.screen_name == ScreenName.INTRO:
            self._go_to(ScreenName.DEMOS)
        elif self.screen_name == ScreenName.DEMO:
            self._open_pause()
        elif self.screen_name == ScreenName.SETTINGS:
            self._go_to(self.settings_return_screen)
        elif self.screen_name == ScreenName.PAUSE:
            self._resume()

    def _open_pause(self) -> None:
        self.previous_screen = self.screen_name
        self._go_to(ScreenName.PAUSE)

    def _resume(self) -> None:
        self._go_to(self.previous_screen)

    def _open_settings(self) -> None:
        if self.screen_name in {ScreenName.DEMO, ScreenName.SETTINGS}:
            return

        self.help_visible = False
        self.settings_return_screen = self.screen_name
        self._go_to(ScreenName.SETTINGS)

    def _toggle_help(self) -> None:
        self.help_visible = not self.help_visible

    def _toggle_language(self) -> None:
        if self.context.settings.language == "pl":
            self.context.settings.language = "en"
        else:
            self.context.settings.language = "pl"

    def _go_to(self, screen_name: ScreenName) -> None:
        self.screen_name = screen_name
        self.selected_index = 0
        self.menu_items = []

    def _handle_scene_command(self, command: SceneCommand) -> None:
        if command.kind == SceneCommandKind.NONE:
            return
        if command.kind == SceneCommandKind.PAUSE:
            self._open_pause()
        elif command.kind == SceneCommandKind.BACK_TO_DEMOS:
            self.scene_manager.clear()
            self._go_to(ScreenName.DEMOS)
        elif command.kind == SceneCommandKind.RESTART:
            self._restart_current_demo()
        elif command.kind == SceneCommandKind.QUIT:
            self.running = False

    def _restart_current_demo(self) -> None:
        demo = self._require_demo()
        if demo.create_scene is None:
            return

        self.scene_manager.replace(demo.create_scene(self.context))

    def _current_menu_item_count(self) -> int:
        counts = {
            ScreenName.LANGUAGE: 2,
            ScreenName.LEVELS: len(levels_from_manifests()),
            ScreenName.DEMOS: len(self._current_level_demos()),
            ScreenName.INTRO: 1,
            ScreenName.DEMO: 1,
            ScreenName.SETTINGS: 5,
            ScreenName.PAUSE: 6,
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

    def _language_label(self) -> str:
        if self.context.settings.language == "pl":
            return "Polski"

        return "English"

    def _on_off(self, enabled: bool) -> str:
        return self._text("On", "Włączone") if enabled else self._text("Off", "Wyłączone")


def main() -> None:
    """Run the unified app shell."""
    UnifiedAppShell().run()


if __name__ == "__main__":
    main()
