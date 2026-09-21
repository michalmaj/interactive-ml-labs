"""Renderer for the unified shell learning progress report."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import BadgeItem, MenuItem


@dataclass(frozen=True, slots=True)
class ProgressReportMetric:
    """One top-level progress metric shown in the report."""

    label: str
    completed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class ProgressReportPath:
    """Progress summary for one guided learning path."""

    title: str
    status_label: str
    lesson_label: str
    task_label: str
    badge_label: str
    completed_count: int
    total_count: int


@dataclass(frozen=True, slots=True)
class ProgressReportDetails:
    """Renderable data for the learning progress report."""

    title: str
    subtitle: str
    metrics: list[ProgressReportMetric]
    explain_heading: str
    explain_lines: list[str]
    paths_heading: str
    paths: list[ProgressReportPath]
    badges_heading: str
    badge_summary: str
    badges: list[BadgeItem]


@dataclass(frozen=True, slots=True)
class ProgressReportFonts:
    """Fonts used by the progress report renderer."""

    title: pygame.font.Font
    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class ProgressReportColors:
    """Resolved colors used by the progress report renderer."""

    background: tuple[int, int, int]
    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]
    selected_panel: tuple[int, int, int]
    border: tuple[int, int, int]
    progress_track: tuple[int, int, int]
    unlocked_fill: tuple[int, int, int]
    unlocked_outline: tuple[int, int, int]
    locked_fill: tuple[int, int, int]
    locked_outline: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class ProgressReportRenderResult:
    """Geometry returned after rendering the progress report."""

    content_end: int
    viewport: pygame.Rect
    menu_items: list[MenuItem]


class ProgressReportRenderer:
    """Draw the shell learning progress report."""

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        details: ProgressReportDetails,
        scroll_offset: int,
        max_scroll: int,
        content_bottom: int,
        footer_y: int,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> ProgressReportRenderResult:
        """Render the report and return layout data for shell scroll state."""
        width, _ = settings.resolution
        content_width = min(1040, width - 160)
        y = 72

        self._draw_text(surface, details.title, (80, y), fonts.title, colors.text)
        y += 58
        y = self._draw_wrapped(
            surface,
            details.subtitle,
            (80, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 24

        report_top = y
        report_bottom = content_bottom - 76
        viewport = pygame.Rect(80, report_top, content_width, max(0, report_bottom - report_top))
        previous_clip = surface.get_clip()
        surface.set_clip(viewport)

        try:
            content_y = report_top - scroll_offset
            content_y = self._draw_metrics(
                surface,
                details.metrics,
                (80, content_y),
                content_width,
                fonts=fonts,
                colors=colors,
            )
            content_y += 24
            content_y = self._draw_section(
                surface,
                details.explain_heading,
                details.explain_lines,
                (80, content_y),
                content_width,
                fonts=fonts,
                colors=colors,
            )
            content_y += 20
            content_y = self._draw_paths(
                surface,
                details.paths_heading,
                details.paths,
                (80, content_y),
                content_width,
                fonts=fonts,
                colors=colors,
            )
            content_y += 20
            content_y = self._draw_badges(
                surface,
                details,
                (80, content_y),
                content_width,
                fonts=fonts,
                colors=colors,
            )
        finally:
            surface.set_clip(previous_clip)

        menu_items = self._draw_menu(
            surface,
            [self._text(settings, "Back", "Wróć")],
            top=content_bottom - 62,
            width=260,
            fonts=fonts,
            colors=colors,
        )
        self._draw_text(
            surface,
            self._text(
                settings,
                "Wheel: scroll | Enter/Esc/Backspace: home | L: language",
                "Kółko: przewijaj | Enter/Esc/Backspace: start | L: język",
            ),
            (80, footer_y),
            fonts.small,
            colors.muted_text,
        )
        _ = max_scroll
        return ProgressReportRenderResult(
            content_end=content_y,
            viewport=viewport,
            menu_items=menu_items,
        )

    def _draw_metrics(
        self,
        surface: pygame.Surface,
        metrics: list[ProgressReportMetric],
        position: tuple[int, int],
        width: int,
        *,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> int:
        """Draw top-level metric cards."""
        x, y = position
        column_count = 2 if width >= 900 else 1
        gap = 20
        column_width = (width - gap) // 2 if column_count == 2 else width
        row_tops = [y for _ in range(column_count)]

        for index, metric in enumerate(metrics):
            column = index % column_count
            card_x = x + column * (column_width + gap)
            card_y = row_tops[column]
            rect = pygame.Rect(card_x, card_y, column_width, 82)
            pygame.draw.rect(surface, colors.panel, rect, border_radius=8)
            pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)
            label_y = self._draw_wrapped(
                surface,
                metric.label,
                (rect.x + 18, rect.y + 14),
                rect.width - 36,
                fonts.small,
                colors.text,
            )
            self._draw_progress_bar(
                surface,
                rect.x + 18,
                label_y + 8,
                rect.width - 36,
                metric.completed_count,
                metric.total_count,
                colors=colors,
            )
            row_tops[column] = rect.bottom + 16

        return max(row_tops) if row_tops else y

    def _draw_section(
        self,
        surface: pygame.Surface,
        heading: str,
        lines: list[str],
        position: tuple[int, int],
        width: int,
        *,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> int:
        """Draw a headed list of report lines."""
        x, y = position
        self._draw_text(surface, heading, (x, y), fonts.heading, colors.text)
        y += 42
        for line in lines:
            y = self._draw_wrapped(surface, f"- {line}", (x, y), width, fonts.body, colors.text)
            y += 8

        return y

    def _draw_paths(
        self,
        surface: pygame.Surface,
        heading: str,
        paths: list[ProgressReportPath],
        position: tuple[int, int],
        width: int,
        *,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> int:
        """Draw progress rows for guided learning paths."""
        x, y = position
        self._draw_text(surface, heading, (x, y), fonts.heading, colors.text)
        y += 48
        for path in paths:
            rect = pygame.Rect(x, y, width, 122)
            pygame.draw.rect(surface, colors.panel, rect, border_radius=8)
            pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)
            content_x = rect.x + 18
            content_y = rect.y + 14
            label_width = rect.width - 36
            content_y = self._draw_wrapped(
                surface,
                path.title,
                (content_x, content_y),
                label_width,
                fonts.body,
                colors.text,
            )
            content_y += 2
            self._draw_text(
                surface, path.status_label, (content_x, content_y), fonts.small, colors.accent
            )
            content_y += 24
            self._draw_progress_bar(
                surface,
                content_x,
                content_y,
                label_width,
                path.completed_count,
                path.total_count,
                colors=colors,
            )
            content_y += 18
            y_after = self._draw_wrapped(
                surface,
                f"{path.lesson_label} | {path.task_label} | {path.badge_label}",
                (content_x, content_y),
                label_width,
                fonts.small,
                colors.muted_text,
            )
            y = max(rect.bottom + 12, y_after + 12)

        return y

    def _draw_badges(
        self,
        surface: pygame.Surface,
        details: ProgressReportDetails,
        position: tuple[int, int],
        width: int,
        *,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> int:
        """Draw unlocked and locked badge labels."""
        x, y = position
        self._draw_text(surface, details.badges_heading, (x, y), fonts.heading, colors.text)
        y += 42
        y = self._draw_wrapped(
            surface,
            details.badge_summary,
            (x, y),
            width,
            fonts.body,
            colors.accent,
        )
        y += 18
        for badge in details.badges:
            icon_y = y + fonts.small.get_linesize() // 2
            self._draw_badge_medallion(
                surface,
                (x + 12, icon_y),
                unlocked=badge.unlocked,
                colors=colors,
            )
            y = self._draw_wrapped(
                surface,
                badge.label,
                (x + 34, y),
                width - 34,
                fonts.small,
                colors.text if badge.unlocked else colors.muted_text,
            )
            y += 6

        return y

    def _draw_menu(
        self,
        surface: pygame.Surface,
        labels: list[str],
        *,
        top: int,
        width: int,
        fonts: ProgressReportFonts,
        colors: ProgressReportColors,
    ) -> list[MenuItem]:
        """Draw a compact shell menu and return hitboxes."""
        menu_items: list[MenuItem] = []
        for index, label in enumerate(labels):
            rect = pygame.Rect(80, top + index * 70, width, 54)
            color = colors.selected_panel if index == 0 else colors.panel
            pygame.draw.rect(surface, color, rect, border_radius=8)
            pygame.draw.rect(surface, colors.border, rect, width=1, border_radius=8)
            label_y = rect.y + max(0, (rect.height - fonts.body.get_height()) // 2)
            self._draw_text(surface, label, (rect.x + 20, label_y), fonts.body, colors.text)
            menu_items.append(MenuItem(label=label, rect=rect))

        return menu_items

    def _draw_progress_bar(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        width: int,
        completed_count: int,
        total_count: int,
        *,
        colors: ProgressReportColors,
    ) -> None:
        """Draw one compact progress bar."""
        track_rect = pygame.Rect(x, y, width, 8)
        pygame.draw.rect(surface, colors.progress_track, track_rect, border_radius=3)
        if total_count <= 0 or completed_count <= 0:
            return

        ratio = min(1.0, completed_count / total_count)
        fill_rect = pygame.Rect(x, y, round(width * ratio), track_rect.height)
        pygame.draw.rect(surface, colors.accent, fill_rect, border_radius=3)

    def _draw_badge_medallion(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        *,
        unlocked: bool,
        colors: ProgressReportColors,
    ) -> None:
        """Draw a small local badge icon."""
        fill = colors.unlocked_fill if unlocked else colors.locked_fill
        outline = colors.unlocked_outline if unlocked else colors.locked_outline
        detail = colors.background if unlocked else colors.muted_text
        cx, cy = center

        pygame.draw.circle(surface, fill, center, 12)
        pygame.draw.circle(surface, outline, center, 12, width=2)

        if unlocked:
            pygame.draw.line(surface, detail, (cx - 5, cy), (cx - 1, cy + 4), width=3)
            pygame.draw.line(surface, detail, (cx - 1, cy + 4), (cx + 6, cy - 5), width=3)
            return

        shackle = pygame.Rect(cx - 5, cy - 6, 10, 9)
        body = pygame.Rect(cx - 6, cy - 1, 12, 8)
        pygame.draw.arc(surface, detail, shackle, 3.14, 6.28, width=2)
        pygame.draw.rect(surface, detail, body, border_radius=2)

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
