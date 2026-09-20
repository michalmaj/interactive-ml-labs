"""Renderer for the unified shell badge gallery."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings
from interactive_ml_labs.shell_types import BadgeItem, MenuItem


@dataclass(frozen=True, slots=True)
class BadgeGalleryPath:
    """Renderable badge data for one guided learning path."""

    title: str
    progress_label: str
    badges: list[BadgeItem]


@dataclass(frozen=True, slots=True)
class BadgeScreenFonts:
    """Fonts used by the badge gallery renderer."""

    title: pygame.font.Font
    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class BadgeScreenColors:
    """Resolved colors used by the badge gallery renderer."""

    background: tuple[int, int, int]
    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    panel: tuple[int, int, int]
    selected_panel: tuple[int, int, int]
    border: tuple[int, int, int]
    unlocked_fill: tuple[int, int, int]
    unlocked_outline: tuple[int, int, int]
    locked_fill: tuple[int, int, int]
    locked_outline: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class BadgeScreenRenderResult:
    """Geometry returned after rendering the badge gallery."""

    content_end: int
    viewport: pygame.Rect
    menu_items: list[MenuItem]


class BadgeGalleryRenderer:
    """Draw the shell badge gallery."""

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        paths: list[BadgeGalleryPath],
        summary_label: str,
        scroll_offset: int,
        content_bottom: int,
        footer_y: int,
        fonts: BadgeScreenFonts,
        colors: BadgeScreenColors,
    ) -> BadgeScreenRenderResult:
        """Render the badge gallery and return layout data for shell scroll state."""
        width, _ = settings.resolution
        content_width = min(1040, width - 160)
        y = 82

        self._draw_text(
            surface,
            self._text(settings, "Badges", "Odznaki"),
            (80, y),
            fonts.title,
            colors.text,
        )
        y += 58
        y = self._draw_wrapped(
            surface,
            self._text(
                settings,
                "Track the concepts you have demonstrated across guided learning paths.",
                "Zobacz, które pojęcia masz już przećwiczone w prowadzonych ścieżkach.",
            ),
            (80, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 24
        y = self._draw_wrapped(
            surface,
            summary_label,
            (80, y),
            content_width,
            fonts.heading,
            colors.accent,
        )
        content_top = y + 26
        gallery_bottom = content_bottom - 76
        content_y = content_top - scroll_offset

        column_count = 2 if width >= 1180 else 1
        column_gap = 48
        column_width = (content_width - column_gap) // 2 if column_count == 2 else content_width
        column_tops = [content_y for _ in range(column_count)]
        viewport = pygame.Rect(80, content_top, content_width, max(0, gallery_bottom - content_top))
        previous_clip = surface.get_clip()
        surface.set_clip(viewport)

        try:
            for index, path in enumerate(paths):
                column = index % column_count
                x = 80 + column * (column_width + column_gap)
                path_y = column_tops[column]
                path_y = self._draw_wrapped_visible(
                    surface,
                    path.title,
                    (x, path_y),
                    column_width,
                    fonts.heading,
                    colors.text,
                    content_top,
                    gallery_bottom,
                )
                path_y += 8
                path_y = self._draw_wrapped_visible(
                    surface,
                    path.progress_label,
                    (x, path_y),
                    column_width,
                    fonts.small,
                    colors.accent,
                    content_top,
                    gallery_bottom,
                )
                path_y += 10
                for badge in path.badges:
                    icon_y = path_y + fonts.small.get_linesize() // 2
                    self._draw_badge_medallion(
                        surface,
                        (x + 12, icon_y),
                        unlocked=badge.unlocked,
                        colors=colors,
                    )
                    path_y = self._draw_wrapped_visible(
                        surface,
                        badge.label,
                        (x + 34, path_y),
                        column_width - 34,
                        fonts.small,
                        colors.text if badge.unlocked else colors.muted_text,
                        content_top,
                        gallery_bottom,
                    )
                    path_y += 4

                column_tops[column] = path_y + 26
        finally:
            surface.set_clip(previous_clip)

        content_end = max(column_tops) if column_tops else content_y
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
        return BadgeScreenRenderResult(
            content_end=content_end,
            viewport=viewport,
            menu_items=menu_items,
        )

    def _draw_menu(
        self,
        surface: pygame.Surface,
        labels: list[str],
        *,
        top: int,
        width: int,
        fonts: BadgeScreenFonts,
        colors: BadgeScreenColors,
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

    def _draw_badge_medallion(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        *,
        unlocked: bool,
        colors: BadgeScreenColors,
    ) -> None:
        """Draw a small local badge icon without external image assets."""
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

            self._draw_text(surface, line, (x, y), font, color)
            y += font.get_linesize()
            line = word

        if line:
            self._draw_text(surface, line, (x, y), font, color)
            y += font.get_linesize()

        return y

    def _draw_wrapped_visible(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        top: int,
        bottom: int,
    ) -> int:
        """Draw wrapped text only when each line intersects the visible viewport."""
        words = text.split()
        line = ""
        x, y = position

        for word in words:
            candidate = f"{line} {word}".strip()
            if font.size(candidate)[0] <= width:
                line = candidate
                continue

            if self._line_is_visible(y, font, top, bottom):
                self._draw_text(surface, line, (x, y), font, color)
            y += font.get_linesize()
            line = word

        if line:
            if self._line_is_visible(y, font, top, bottom):
                self._draw_text(surface, line, (x, y), font, color)
            y += font.get_linesize()

        return y

    @staticmethod
    def _line_is_visible(y: int, font: pygame.font.Font, top: int, bottom: int) -> bool:
        """Return whether one text line fits fully inside the visible viewport."""
        return y >= top and y + font.get_linesize() <= bottom

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
