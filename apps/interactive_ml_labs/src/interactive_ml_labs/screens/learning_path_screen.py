"""Renderer for the guided learning paths screen."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import MenuItem


@dataclass(frozen=True, slots=True)
class LearningPathMetric:
    """One progress metric displayed in the learning path details panel."""

    label: str
    completed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class LearningPathDetails:
    """Renderable details for one guided learning path."""

    title: str
    summary: str
    lesson_count_label: str
    progress_metrics: list[LearningPathMetric]
    status_label: str
    next_action_label: str
    badge_labels: list[str]
    course_map_heading: str
    lesson_labels: list[str]


@dataclass(frozen=True, slots=True)
class LearningPathScreenFonts:
    """Fonts used by the guided learning paths renderer."""

    title: pygame.font.Font
    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class LearningPathScreenColors:
    """Resolved colors used by the guided learning paths renderer."""

    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]
    selected_panel: tuple[int, int, int]
    border: tuple[int, int, int]
    progress_track: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class LearningPathRenderResult:
    """Geometry returned after rendering guided learning paths."""

    content_end: int
    viewport: pygame.Rect
    menu_items: list[MenuItem]


class LearningPathRenderer:
    """Draw the guided learning paths screen."""

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        selected_index: int,
        menu_labels: list[str],
        details: LearningPathDetails,
        scroll_offset: int,
        max_scroll: int,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
        footer_y: int,
    ) -> LearningPathRenderResult:
        """Render guided learning paths and return layout data for shell state."""
        self._draw_title(
            surface,
            self._text(settings, "Guided learning paths", "Prowadzone ścieżki nauki"),
            self._text(
                settings,
                "Follow lessons that build one idea at a time.",
                "Przechodź lekcje, które budują intuicję krok po kroku.",
            ),
            fonts=fonts,
            colors=colors,
        )
        menu_items = self._draw_menu(
            surface,
            menu_labels,
            selected_index=selected_index,
            top=210,
            width=520,
            fonts=fonts,
            colors=colors,
        )
        content_end, viewport = self._draw_details_panel(
            surface,
            settings=settings,
            details=details,
            scroll_offset=scroll_offset,
            max_scroll=max_scroll,
            fonts=fonts,
            colors=colors,
        )
        self._draw_text(
            surface,
            self._text(
                settings,
                "Enter: lessons | Esc/Backspace: home | S: settings | L: language",
                "Enter: lekcje | Esc/Backspace: start | S: ustawienia | L: język",
            ),
            (80, footer_y),
            fonts.small,
            colors.muted_text,
        )
        return LearningPathRenderResult(
            content_end=content_end,
            viewport=viewport,
            menu_items=menu_items,
        )

    def _draw_title(
        self,
        surface: pygame.Surface,
        title: str,
        subtitle: str,
        *,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
    ) -> None:
        """Draw the screen title block."""
        self._draw_text(surface, title, (80, 70), fonts.title, colors.text)
        self._draw_text(surface, subtitle, (82, 128), fonts.body, colors.muted_text)

    def _draw_menu(
        self,
        surface: pygame.Surface,
        labels: list[str],
        *,
        selected_index: int,
        top: int,
        width: int,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
    ) -> list[MenuItem]:
        """Draw the learning path menu and return hitboxes."""
        menu_items: list[MenuItem] = []
        for index, label in enumerate(labels):
            rect = pygame.Rect(80, top + index * 70, width, 54)
            color = colors.selected_panel if index == selected_index else colors.panel
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)
            label_y = rect.y + max(0, (rect.height - fonts.body.get_height()) // 2)
            self._draw_text(surface, label, (rect.x + 20, label_y), fonts.body, colors.text)
            menu_items.append(MenuItem(label=label, rect=rect))

        return menu_items

    def _draw_details_panel(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        details: LearningPathDetails,
        scroll_offset: int,
        max_scroll: int,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
    ) -> tuple[int, pygame.Rect]:
        """Draw the right-hand learning path details panel."""
        width, height = settings.resolution
        left = 660
        top = 190
        panel_width = max(360, width - left - 80)
        panel_height = max(320, height - top - 100)
        rect = pygame.Rect(left, top, panel_width, panel_height)

        pygame.draw.rect(surface, colors.panel, rect, border_radius=8)
        pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)

        viewport = rect.inflate(-28, -28)
        scrollbar_margin = 14 if max_scroll > 0 else 0
        previous_clip = surface.get_clip()
        surface.set_clip(viewport)

        try:
            x = rect.x + 28
            y = rect.y + 28 - scroll_offset
            content_width = rect.width - 56 - scrollbar_margin
            y = self._draw_details_content(
                surface,
                details,
                (x, y),
                content_width,
                fonts=fonts,
                colors=colors,
            )
        finally:
            surface.set_clip(previous_clip)

        return y, viewport

    def _draw_details_content(
        self,
        surface: pygame.Surface,
        details: LearningPathDetails,
        position: tuple[int, int],
        content_width: int,
        *,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
    ) -> int:
        """Draw learning path details and return the next y coordinate."""
        x, y = position
        y = self._draw_wrapped(
            surface,
            details.title,
            (x, y),
            content_width,
            fonts.heading,
            colors.text,
        )
        y += 12
        y = self._draw_wrapped(
            surface,
            details.summary,
            (x, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 22
        self._draw_text(surface, details.lesson_count_label, (x, y), fonts.small, colors.accent)
        y += 28
        for metric in details.progress_metrics:
            y = self._draw_metric(surface, metric, x, y, content_width, fonts=fonts, colors=colors)

        y = self._draw_wrapped(
            surface,
            details.status_label,
            (x, y),
            content_width,
            fonts.small,
            colors.accent,
        )
        y += 6
        y = self._draw_wrapped(
            surface,
            details.next_action_label,
            (x, y),
            content_width,
            fonts.small,
            colors.accent,
        )
        y += 18
        for badge_label in details.badge_labels:
            y = self._draw_wrapped(
                surface,
                badge_label,
                (x, y),
                content_width,
                fonts.small,
                colors.muted_text,
            )
            y += 4

        y += 16
        self._draw_text(surface, details.course_map_heading, (x, y), fonts.small, colors.accent)
        y += 28
        for lesson_label in details.lesson_labels:
            y = self._draw_wrapped(
                surface,
                lesson_label,
                (x, y),
                content_width,
                fonts.small,
                colors.text,
            )
            y += 4

        return y

    def _draw_metric(
        self,
        surface: pygame.Surface,
        metric: LearningPathMetric,
        x: int,
        y: int,
        width: int,
        *,
        fonts: LearningPathScreenFonts,
        colors: LearningPathScreenColors,
    ) -> int:
        """Draw one metric label and compact progress bar."""
        y = self._draw_wrapped(surface, metric.label, (x, y), width, fonts.small, colors.text)
        self._draw_compact_progress_bar(
            surface,
            x,
            y + 2,
            width,
            metric.completed_count,
            metric.total_count,
            colors=colors,
        )
        return y + 20

    def _draw_compact_progress_bar(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        width: int,
        completed_count: int,
        total_count: int,
        *,
        colors: LearningPathScreenColors,
    ) -> None:
        """Draw one compact progress bar."""
        track_rect = pygame.Rect(x, y, width, 8)
        pygame.draw.rect(surface, colors.progress_track, track_rect, border_radius=3)
        if total_count <= 0 or completed_count <= 0:
            return

        ratio = min(1.0, completed_count / total_count)
        fill_rect = pygame.Rect(x, y, round(width * ratio), track_rect.height)
        pygame.draw.rect(surface, colors.accent, fill_rect, border_radius=3)

    def _draw_wrapped(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> int:
        """Draw wrapped text and return the next y coordinate."""
        words = text.split()
        line = ""
        x, y = position
        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width:
                line = candidate
                continue

            if line:
                self._draw_text(surface, line, (x, y), font, color)
                y += font.get_linesize()
            line = word

        if line:
            self._draw_text(surface, line, (x, y), font, color)
            y += font.get_linesize()

        return y

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
