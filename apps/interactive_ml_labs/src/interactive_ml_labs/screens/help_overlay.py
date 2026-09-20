"""Renderer for the unified shell help overlay."""

from __future__ import annotations

from dataclasses import dataclass

import pygame

from interactive_ml_labs.settings import AppSettings


@dataclass(frozen=True, slots=True)
class HelpOverlayDetails:
    """Renderable help overlay content."""

    title: str
    summary: str | None
    objectives: list[str]
    controls: list[str]
    fallback_body: str
    goals_heading: str
    controls_heading: str


@dataclass(frozen=True, slots=True)
class HelpOverlayFonts:
    """Fonts used by the help overlay renderer."""

    heading: pygame.font.Font
    body: pygame.font.Font
    small: pygame.font.Font


@dataclass(frozen=True, slots=True)
class HelpOverlayColors:
    """Resolved colors used by the help overlay renderer."""

    text: tuple[int, int, int]
    muted_text: tuple[int, int, int]
    accent: tuple[int, int, int]
    background: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class HelpOverlayRenderResult:
    """Geometry returned after rendering the help overlay."""

    rect: pygame.Rect
    content_bottom: int


class HelpOverlayRenderer:
    """Draw the shared help overlay."""

    def render(
        self,
        surface: pygame.Surface,
        *,
        settings: AppSettings,
        details: HelpOverlayDetails,
        fonts: HelpOverlayFonts,
        colors: HelpOverlayColors,
    ) -> HelpOverlayRenderResult:
        """Render the help overlay."""
        width, height = settings.resolution
        margin_x = min(120, max(32, width // 12))
        margin_y = min(90, max(32, height // 10))
        rect = pygame.Rect(
            margin_x,
            margin_y,
            width - 2 * margin_x,
            height - 2 * margin_y,
        )
        pygame.draw.rect(surface, colors.background, rect, border_radius=8)
        pygame.draw.rect(surface, colors.accent, rect, width=2, border_radius=8)

        self._draw_text(
            surface,
            details.title,
            (rect.x + 32, rect.y + 28),
            fonts.heading,
            colors.text,
        )

        y = rect.y + 78
        content_width = rect.width - 64
        if details.summary is None:
            y = self._draw_wrapped(
                surface,
                details.fallback_body,
                (rect.x + 32, y),
                content_width,
                fonts.body,
                colors.muted_text,
            )
            return HelpOverlayRenderResult(rect=rect, content_bottom=y)

        y = self._draw_wrapped(
            surface,
            details.summary,
            (rect.x + 32, y),
            content_width,
            fonts.body,
            colors.muted_text,
        )
        y += 22

        if rect.width >= 880:
            gap = 36
            column_width = (content_width - gap) // 2
            goals_bottom = self._draw_help_section(
                surface,
                details.goals_heading,
                details.objectives,
                rect.x + 32,
                y,
                column_width,
                fonts=fonts,
                colors=colors,
            )
            controls_bottom = self._draw_help_section(
                surface,
                details.controls_heading,
                details.controls,
                rect.x + 32 + column_width + gap,
                y,
                column_width,
                fonts=fonts,
                colors=colors,
            )
            return HelpOverlayRenderResult(
                rect=rect,
                content_bottom=max(goals_bottom, controls_bottom),
            )

        y = self._draw_help_section(
            surface,
            details.goals_heading,
            details.objectives,
            rect.x + 32,
            y,
            content_width,
            fonts=fonts,
            colors=colors,
        )
        y += 12
        y = self._draw_help_section(
            surface,
            details.controls_heading,
            details.controls,
            rect.x + 32,
            y,
            content_width,
            fonts=fonts,
            colors=colors,
        )
        return HelpOverlayRenderResult(rect=rect, content_bottom=y)

    def _draw_help_section(
        self,
        surface: pygame.Surface,
        title: str,
        items: list[str],
        x: int,
        y: int,
        width: int,
        *,
        fonts: HelpOverlayFonts,
        colors: HelpOverlayColors,
    ) -> int:
        """Draw one help section and return the next y coordinate."""
        self._draw_text(surface, title, (x, y), fonts.small, colors.accent)
        y += 28
        for item in items:
            y = self._draw_wrapped(
                surface,
                f"- {item}",
                (x + 16, y),
                width - 16,
                fonts.small,
                colors.text,
            )
            y += 4

        return y

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
