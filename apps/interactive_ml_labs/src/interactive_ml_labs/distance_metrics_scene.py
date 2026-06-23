"""Native Distance Metrics Lab scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.readout_panel import (
    ReadoutPanelColors,
    ReadoutPanelFonts,
    draw_readout_panel,
)
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext
from interactive_ml_labs.ui_helpers import draw_panel, draw_text, draw_wrapped_text

BACKGROUND: Final[tuple[int, int, int]] = (20, 23, 27)
PANEL: Final[tuple[int, int, int]] = (34, 39, 45)
PLOT_BG: Final[tuple[int, int, int]] = (16, 19, 23)
GRID: Final[tuple[int, int, int]] = (48, 55, 63)
TEXT: Final[tuple[int, int, int]] = (236, 239, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (166, 174, 184)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
SECONDARY: Final[tuple[int, int, int]] = (248, 183, 96)
GOOD: Final[tuple[int, int, int]] = (147, 218, 155)
WARNING: Final[tuple[int, int, int]] = (246, 132, 134)

QUERY_STEP: Final[float] = 0.35


class DistanceMetric(Enum):
    """Supported distance metrics."""

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """Labeled point in a toy feature space."""

    x: float
    y: float
    label: str


@dataclass(frozen=True, slots=True)
class DistancePreset:
    """Static distance metric scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    query: tuple[float, float]
    points: tuple[MetricPoint, ...]

    def name_for_language(self, language: str) -> str:
        """Return localized preset name."""
        if language == "pl":
            return self.name_pl
        return self.name_en

    def summary_for_language(self, language: str) -> str:
        """Return localized summary."""
        if language == "pl":
            return self.summary_pl
        return self.summary_en


PRESETS: Final[tuple[DistancePreset, ...]] = (
    DistancePreset(
        name_en="Diagonal trade-off",
        name_pl="Diagonalny kompromis",
        summary_en="Euclidean rewards diagonal closeness, while Manhattan counts axis steps.",
        summary_pl="Euclidean docenia bliskość po skosie, a Manhattan liczy kroki po osiach.",
        query=(-0.2, 0.2),
        points=(
            MetricPoint(-2.2, -1.7, "A"),
            MetricPoint(-1.2, 1.6, "B"),
            MetricPoint(1.4, 0.8, "C"),
            MetricPoint(2.0, -1.1, "D"),
        ),
    ),
    DistancePreset(
        name_en="Axis mismatch",
        name_pl="Różnica po jednej osi",
        summary_en="A point can look close on one feature and far on another feature.",
        summary_pl="Punkt może być blisko po jednej cesze i daleko po drugiej.",
        query=(0.4, -0.4),
        points=(
            MetricPoint(-1.8, -0.6, "A"),
            MetricPoint(0.8, 2.1, "B"),
            MetricPoint(2.3, -0.2, "C"),
            MetricPoint(-0.7, 1.0, "D"),
        ),
    ),
    DistancePreset(
        name_en="Cluster edge",
        name_pl="Krawędź klastra",
        summary_en="Near a cluster edge, the chosen metric can change the nearest point.",
        summary_pl="Na krawędzi klastra wybrana metryka może zmienić najbliższy punkt.",
        query=(1.0, 0.9),
        points=(
            MetricPoint(-1.5, -1.6, "A"),
            MetricPoint(0.4, 2.3, "B"),
            MetricPoint(1.9, -0.3, "C"),
            MetricPoint(2.6, 1.8, "D"),
        ),
    ),
)

METRICS: Final[tuple[DistanceMetric, ...]] = (
    DistanceMetric.EUCLIDEAN,
    DistanceMetric.MANHATTAN,
    DistanceMetric.CHEBYSHEV,
)


class DistanceMetricsLabScene:
    """Interactive slice for comparing nearest neighbors under distance metrics."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic distance metric scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.metric_index = 0
        self.query_x, self.query_y = PRESETS[0].query

    @property
    def preset(self) -> DistancePreset:
        """Return the active distance preset."""
        return PRESETS[self.preset_index]

    @property
    def metric(self) -> DistanceMetric:
        """Return the active distance metric."""
        return METRICS[self.metric_index]

    def handle_event(self, event: object) -> SceneCommand:
        """Handle one input event."""
        if isinstance(event, pygame.event.Event) and event.type == pygame.KEYDOWN:
            self._handle_keydown(event.key)
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance scene state."""
        _ = dt
        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the distance metrics lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_space_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self._reset_query()
        elif key == pygame.K_m:
            self.metric_index = (self.metric_index + 1) % len(METRICS)
        elif key == pygame.K_LEFT:
            self.query_x = max(-3.0, self.query_x - QUERY_STEP)
        elif key == pygame.K_RIGHT:
            self.query_x = min(3.0, self.query_x + QUERY_STEP)
        elif key == pygame.K_DOWN:
            self.query_y = max(-3.0, self.query_y - QUERY_STEP)
        elif key == pygame.K_UP:
            self.query_y = min(3.0, self.query_y + QUERY_STEP)
        elif key == pygame.K_r:
            self.preset_index = 0
            self.metric_index = 0
            self._reset_query()

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Distance Metrics Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Move a query point and see what nearest really means.",
                "Przesuwaj query point i zobacz, co naprawdę znaczy nearest.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_space_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Feature space", "Feature space"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._metric_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        plot_rect = pygame.Rect(rect.x + 88, rect.y + 104, 520, 270)
        self._draw_feature_space(surface, plot_rect)
        self._draw_distance_bars(surface, pygame.Rect(rect.x + 88, rect.y + 404, 520, 42))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 44),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        nearest = self._nearest_point()
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("metric", "metryka"), self._metric_label()),
            (self._label("query x", "query x"), f"{self.query_x:+.1f}"),
            (self._label("query y", "query y"), f"{self.query_y:+.1f}"),
            (self._label("nearest", "nearest"), nearest.label),
            (self._label("distance", "distance"), f"{self._distance_to(nearest):.2f}"),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Nearest readout", "Odczyt nearest"),
            rows=rows,
            options=options,
            takeaway=self._active_takeaway(),
            fonts=ReadoutPanelFonts(heading=self._font_heading, small=self._font_small),
            colors=ReadoutPanelColors(
                panel=PANEL,
                text=TEXT,
                muted_text=MUTED_TEXT,
                accent=ACCENT,
                secondary=SECONDARY,
            ),
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: understand why k-NN depends on the distance you choose.",
                "Cel: zrozum, czemu k-NN zależy od wybranej metryki distance.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_feature_space(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

        query_position = self._to_screen(rect, self.query_x, self.query_y)
        nearest = self._nearest_point()
        nearest_position = self._to_screen(rect, nearest.x, nearest.y)
        pygame.draw.line(surface, SECONDARY, query_position, nearest_position, 2)

        for point in self.preset.points:
            color = GOOD if point is nearest else TEXT
            point_position = self._to_screen(rect, point.x, point.y)
            pygame.draw.circle(surface, color, point_position, 7)
            self._draw_text(
                surface,
                point.label,
                (point_position[0] + 9, point_position[1] - 9),
                self._font_small,
                color,
            )

        pygame.draw.circle(surface, ACCENT, query_position, 8)
        self._draw_text(
            surface,
            "query",
            (query_position[0] + 10, query_position[1] + 8),
            self._font_small,
            ACCENT,
        )

    def _draw_distance_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        distances = tuple((point, self._distance_to(point)) for point in self.preset.points)
        max_distance = max(distance for _, distance in distances)
        bar_width = rect.width // len(distances)
        nearest = self._nearest_point()
        for index, (point, distance) in enumerate(distances):
            x = rect.x + (index * bar_width)
            width = round((distance / max_distance) * (bar_width - 22))
            color = GOOD if point is nearest else MUTED_TEXT
            self._draw_text(surface, point.label, (x, rect.y), self._font_small, color)
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(x + 20, rect.y + 3, width, 12),
                border_radius=4,
            )
            self._draw_text(
                surface, f"{distance:.1f}", (x + 20, rect.y + 21), self._font_small, color
            )

    def _to_screen(self, rect: pygame.Rect, x_value: float, y_value: float) -> tuple[int, int]:
        x = rect.x + round((x_value + 3.2) / 6.4 * rect.width)
        y = rect.bottom - round((y_value + 3.2) / 6.4 * rect.height)
        return (x, y)

    def _distance_to(self, point: MetricPoint) -> float:
        dx = abs(self.query_x - point.x)
        dy = abs(self.query_y - point.y)
        if self.metric is DistanceMetric.MANHATTAN:
            return dx + dy
        if self.metric is DistanceMetric.CHEBYSHEV:
            return max(dx, dy)
        return sqrt((dx * dx) + (dy * dy))

    def _nearest_point(self) -> MetricPoint:
        return min(self.preset.points, key=self._distance_to)

    def _metric_label(self) -> str:
        if self.metric is DistanceMetric.EUCLIDEAN:
            return "Euclidean"
        if self.metric is DistanceMetric.MANHATTAN:
            return "Manhattan"
        return "Chebyshev"

    def _diagnosis_label(self) -> str:
        if self._nearest_point().label != self._nearest_by_euclidean().label:
            return self._label("metric changed nearest", "metryka zmieniła nearest")
        return self._label("same nearest", "ten sam nearest")

    def _active_takeaway(self) -> str:
        if self.metric is DistanceMetric.MANHATTAN:
            return self._label(
                "Manhattan distance counts horizontal plus vertical steps.",
                "Manhattan distance liczy kroki poziome plus pionowe.",
            )
        if self.metric is DistanceMetric.CHEBYSHEV:
            return self._label(
                "Chebyshev distance cares about the largest single-axis gap.",
                "Chebyshev distance patrzy na największą różnicę na jednej osi.",
            )
        return self._label(
            "Euclidean distance measures straight-line closeness.",
            "Euclidean distance mierzy bliskość po linii prostej.",
        )

    def _nearest_by_euclidean(self) -> MetricPoint:
        query_x = self.query_x
        query_y = self.query_y
        return min(
            self.preset.points,
            key=lambda point: sqrt(((query_x - point.x) ** 2) + ((query_y - point.y) ** 2)),
        )

    def _reset_query(self) -> None:
        self.query_x, self.query_y = self.preset.query

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        draw_panel(surface, rect, PANEL)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        draw_text(surface, text, position, font, color)

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
        draw_wrapped_text(
            surface,
            text,
            position,
            width,
            font,
            color,
            line_height=line_height,
        )

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl
        return en


def create_distance_metrics_lab_scene(context: AppContext) -> DistanceMetricsLabScene:
    """Create the unified shell Distance Metrics Lab scene."""
    return DistanceMetricsLabScene(context)
