"""Renderer for the unified shell course map screen."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import MenuItem


@dataclass(frozen=True, slots=True)
class CourseMapMetric:
    """One progress metric displayed in the course map details panel."""

    label: str
    completed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class CourseMapStepDetails:
    """Renderable details for one recommended course-map step."""

    step_label: str
    title: str
    rationale: str
    progress_metrics: list[CourseMapMetric]
    next_action_label: str
    next_reason_label: str
    practice_heading: str
    practice_items: list[str]


@dataclass(frozen=True, slots=True)
class CourseMapOverviewDetails:
    """Renderable details for the free-exploration course-map overview."""

    title: str
    body: str
    progress_metrics: list[CourseMapMetric]


@dataclass(frozen=True, slots=True)
class CourseMapScreenFonts:
    """Fonts used by the course map renderer."""

    title: pygame.font.Font
    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class CourseMapScreenColors:
    """Resolved colors used by the course map renderer."""

    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]
    selected_panel: tuple[int, int, int]
    border: tuple[int, int, int]
    progress_track: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class CourseMapRenderResult:
    """Geometry returned after rendering the course map."""

    content_end: int
    viewport: pygame.Rect
    menu_items: list[MenuItem]


class CourseMapRenderer:
    """Draw the course map screen."""

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        selected_index: int,
        menu_labels: list[str],
        details: CourseMapStepDetails | CourseMapOverviewDetails,
        scroll_offset: int,
        max_scroll: int,
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
        footer_y: int,
    ) -> CourseMapRenderResult:
        """Render the course map and return layout data for shell state."""
        self._draw_title(
            surface,
            self._text(settings, "Course map", "Mapa kursu"),
            self._text(
                settings,
                "Start here, then follow the next idea when it makes sense.",
                "Zacznij tutaj, a potem przechodź do kolejnych intuicji.",
            ),
            fonts=fonts,
            colors=colors,
        )
        menu_items = self._draw_menu(
            surface,
            menu_labels,
            selected_index=selected_index,
            top=170,
            width=560,
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
                "Enter: open path | Esc/Backspace: home | S: settings | L: language",
                "Enter: otwórz ścieżkę | Esc/Backspace: start | S: ustawienia | L: język",
            ),
            (80, footer_y),
            fonts.small,
            colors.muted_text,
        )
        return CourseMapRenderResult(
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
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
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
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
    ) -> list[MenuItem]:
        """Draw the course-map menu and return hitboxes."""
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
        details: CourseMapStepDetails | CourseMapOverviewDetails,
        scroll_offset: int,
        max_scroll: int,
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
    ) -> tuple[int, pygame.Rect]:
        """Draw the right-hand course-map details panel."""
        width, height = settings.resolution
        left = 680
        top = 170
        panel_width = max(360, width - left - 80)
        panel_height = max(380, height - top - 100)
        rect = pygame.Rect(left, top, panel_width, panel_height)

        pygame.draw.rect(surface, colors.panel, rect, border_radius=8)
        pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)

        viewport = rect.inflate(-28, -28)
        scrollbar_margin = 14 if max_scroll > 0 else 0
        previous_clip = surface.get_clip()
        surface.set_clip(viewport)

        try:
            x = rect.x + 28
            y = rect.y + 26 - scroll_offset
            content_width = rect.width - 56 - scrollbar_margin
            if isinstance(details, CourseMapStepDetails):
                y = self._draw_step_details(
                    surface,
                    details,
                    (x, y),
                    content_width,
                    fonts=fonts,
                    colors=colors,
                )
            else:
                y = self._draw_overview_details(
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

    def _draw_step_details(
        self,
        surface: pygame.Surface,
        details: CourseMapStepDetails,
        position: tuple[int, int],
        content_width: int,
        *,
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
    ) -> int:
        """Draw the selected course-map step details."""
        x, y = position
        self._draw_text(surface, details.step_label, (x, y), fonts.small, colors.accent)
        y += 28
        y = self._draw_wrapped(
            surface, details.title, (x, y), content_width, fonts.heading, colors.text
        )
        y += 12
        y = self._draw_wrapped(
            surface,
            details.rationale,
            (x, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 20
        for metric in details.progress_metrics:
            y = self._draw_metric(surface, metric, x, y, content_width, fonts=fonts, colors=colors)

        y += 4
        y = self._draw_wrapped(
            surface,
            details.next_action_label,
            (x, y),
            content_width,
            fonts.small,
            colors.accent,
        )
        y += 18
        y = self._draw_wrapped(
            surface,
            details.next_reason_label,
            (x, y),
            content_width,
            fonts.small,
            colors.muted_text,
        )

        y += 20
        self._draw_text(surface, details.practice_heading, (x, y), fonts.small, colors.accent)
        y += 28
        for item in details.practice_items:
            y = self._draw_wrapped(
                surface, "- " + item, (x + 18, y), content_width - 18, fonts.small, colors.text
            )
            y += 4

        return y

    def _draw_overview_details(
        self,
        surface: pygame.Surface,
        details: CourseMapOverviewDetails,
        position: tuple[int, int],
        content_width: int,
        *,
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
    ) -> int:
        """Draw the full-path-browser overview details."""
        x, y = position
        y = self._draw_wrapped(
            surface, details.title, (x, y), content_width, fonts.heading, colors.text
        )
        y += 12
        y = self._draw_wrapped(
            surface,
            details.body,
            (x, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 22
        for metric in details.progress_metrics:
            y = self._draw_metric(surface, metric, x, y, content_width, fonts=fonts, colors=colors)

        return y

    def _draw_metric(
        self,
        surface: pygame.Surface,
        metric: CourseMapMetric,
        x: int,
        y: int,
        width: int,
        *,
        fonts: CourseMapScreenFonts,
        colors: CourseMapScreenColors,
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
        colors: CourseMapScreenColors,
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
