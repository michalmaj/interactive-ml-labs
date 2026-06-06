"""Native PCA Lab skeleton scene for the unified shell."""

from __future__ import annotations

import math
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BACKGROUND: Final[tuple[int, int, int]] = (21, 24, 28)
PANEL: Final[tuple[int, int, int]] = (34, 39, 45)
PLOT_BG: Final[tuple[int, int, int]] = (17, 20, 24)
GRID: Final[tuple[int, int, int]] = (48, 54, 61)
TEXT: Final[tuple[int, int, int]] = (235, 238, 241)
MUTED_TEXT: Final[tuple[int, int, int]] = (165, 172, 181)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
SECONDARY: Final[tuple[int, int, int]] = (246, 181, 111)
POINT: Final[tuple[int, int, int]] = (146, 217, 150)


class PCALabScene:
    """Static first slice for the PCA / dimensionality reduction lab."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic PCA preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self._points = self._build_preview_points()

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        _ = event
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the PCA preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_original_space(surface, pygame.Rect(52, 138, 520, 420))
        self._draw_projection(surface, pygame.Rect(620, 138, 300, 420))
        self._draw_explained_variance(surface, pygame.Rect(960, 138, 260, 420))
        self._draw_footer(surface)

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "PCA Lab", (52, 42), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "First slice: compare original structure with a one-component projection.",
                "Pierwszy slice: porównaj strukturę danych z projekcją na jedną komponentę.",
            ),
            (52, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_original_space(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Original feature space", "Oryginalna przestrzeń cech"),
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        plot_rect = pygame.Rect(rect.x + 34, rect.y + 72, rect.width - 68, rect.height - 112)
        self._draw_grid(surface, plot_rect)
        axis_start = self._to_screen((-0.72, -0.44), plot_rect)
        axis_end = self._to_screen((0.78, 0.48), plot_rect)
        pygame.draw.line(surface, ACCENT, axis_start, axis_end, 4)
        for point in self._points:
            pygame.draw.circle(surface, POINT, self._to_screen(point, plot_rect), 5)
        self._draw_text(
            surface,
            self._label("principal direction", "główny kierunek"),
            (plot_rect.x + 14, plot_rect.bottom + 16),
            self._font_small,
            ACCENT,
        )

    def _draw_projection(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("1D projection", "Projekcja 1D"),
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        track = pygame.Rect(rect.x + 38, rect.y + 230, rect.width - 76, 8)
        pygame.draw.rect(surface, GRID, track, border_radius=4)
        sorted_scores = sorted(self._projection_score(point) for point in self._points)
        minimum = sorted_scores[0]
        maximum = sorted_scores[-1]
        span = max(maximum - minimum, 0.001)
        for score in sorted_scores:
            x = track.left + round((score - minimum) * track.width / span)
            pygame.draw.circle(surface, POINT, (x, track.centery), 6)
        self._draw_wrapped(
            surface,
            self._label(
                "PCA keeps the direction with the largest spread and drops the rest.",
                "PCA zostawia kierunek z największym rozrzutem i odrzuca resztę.",
            ),
            (rect.x + 22, rect.y + 290),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=20,
        )

    def _draw_explained_variance(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            "Explained variance",
            (rect.x + 22, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        bars = ((0.74, ACCENT, "PC1"), (0.18, SECONDARY, "PC2"), (0.08, MUTED_TEXT, "rest"))
        y = rect.y + 90
        for value, color, label in bars:
            self._draw_text(surface, label, (rect.x + 22, y + 4), self._font_small, TEXT)
            bar_rect = pygame.Rect(rect.x + 76, y, round((rect.width - 120) * value), 24)
            pygame.draw.rect(surface, color, bar_rect, border_radius=5)
            self._draw_text(
                surface,
                f"{round(value * 100)}%",
                (rect.right - 58, y + 4),
                self._font_small,
                MUTED_TEXT,
            )
            y += 52
        self._draw_wrapped(
            surface,
            self._label(
                "Next PR will turn this preview into an interactive projection lab.",
                "Kolejny PR zamieni ten podgląd w interaktywne demo projekcji.",
            ),
            (rect.x + 22, rect.y + 292),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=20,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: see what PCA preserves and why fewer dimensions cost information.",
                "Cel: zobacz, co PCA zachowuje i jaki jest koszt mniejszej liczby wymiarów.",
            ),
            (52, 616),
            self._font_body,
            TEXT,
        )

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.left + round(step * rect.width / 4)
            y = rect.top + round(step * rect.height / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        surface.blit(font.render(text, True, color), position)

    def _draw_wrapped(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        width: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        *,
        line_height: int,
    ) -> None:
        x, y = position
        current = ""
        for word in text.split():
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
                continue

            if current:
                self._draw_text(surface, current, (x, y), font, color)
                y += line_height
            current = word

        if current:
            self._draw_text(surface, current, (x, y), font, color)

    def _to_screen(self, point: tuple[float, float], rect: pygame.Rect) -> tuple[int, int]:
        x, y = point
        return (
            rect.left + round((x + 1.0) * 0.5 * rect.width),
            rect.top + round((1.0 - (y + 1.0) * 0.5) * rect.height),
        )

    def _projection_score(self, point: tuple[float, float]) -> float:
        x, y = point
        return 0.86 * x + 0.51 * y

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl

        return en

    def _build_preview_points(self) -> tuple[tuple[float, float], ...]:
        points: list[tuple[float, float]] = []
        for index in range(36):
            t = -0.92 + index * 1.84 / 35
            wobble = 0.12 * math.sin(index * 1.7)
            points.append((t, 0.56 * t + wobble))
        return tuple(points)


def create_pca_lab_scene(context: AppContext) -> PCALabScene:
    """Create the unified shell PCA Lab scene."""
    return PCALabScene(context)
