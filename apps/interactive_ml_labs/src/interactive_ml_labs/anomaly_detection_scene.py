"""Native Anomaly Detection Lab scene for the unified shell."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
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

THRESHOLDS: Final[tuple[float, ...]] = (0.45, 0.60, 0.75)
DEFAULT_THRESHOLD_INDEX: Final[int] = 1


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in normalized plot coordinates."""

    x: float
    y: float
    true_anomaly: bool


@dataclass(frozen=True, slots=True)
class AnomalyPreset:
    """Static anomaly detection scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    normal_center: tuple[float, float]
    normal_spread: tuple[float, float]
    anomaly_centers: tuple[tuple[float, float], ...]
    normal_count: int
    anomaly_count: int

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


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Confusion counts for an anomaly threshold."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


PRESETS: Final[tuple[AnomalyPreset, ...]] = (
    AnomalyPreset(
        name_en="Payment spikes",
        name_pl="Skoki płatności",
        summary_en="Most records form one dense cloud, with anomalies far from the normal pattern.",
        summary_pl="Większość rekordów tworzy gęstą chmurę, a anomalie leżą daleko od wzorca.",
        normal_center=(-0.18, -0.08),
        normal_spread=(0.26, 0.2),
        anomaly_centers=((0.72, 0.58), (0.64, -0.62), (-0.78, 0.58)),
        normal_count=74,
        anomaly_count=9,
    ),
    AnomalyPreset(
        name_en="Sensor drift",
        name_pl="Sensor drift",
        summary_en="Some anomalies sit near the edge, so a strict threshold can miss them.",
        summary_pl="Część anomalii leży przy brzegu, więc surowy threshold może je przeoczyć.",
        normal_center=(0.04, 0.02),
        normal_spread=(0.34, 0.24),
        anomaly_centers=((0.62, 0.42), (0.8, -0.18), (-0.66, -0.52)),
        normal_count=78,
        anomaly_count=12,
    ),
    AnomalyPreset(
        name_en="User behavior",
        name_pl="Zachowanie użytkowników",
        summary_en="Normal behavior is wider, so a loose threshold creates noisy alerts.",
        summary_pl="Normalne zachowanie jest szersze, więc luźny threshold tworzy szum alertów.",
        normal_center=(-0.04, 0.04),
        normal_spread=(0.42, 0.3),
        anomaly_centers=((0.76, 0.7), (-0.84, -0.62), (0.72, -0.58)),
        normal_count=82,
        anomaly_count=10,
    ),
)


class AnomalyDetectionLabScene:
    """Interactive anomaly score threshold lab for Level 2."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic anomaly detection scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.threshold_index = DEFAULT_THRESHOLD_INDEX
        self.show_scores = True
        self._rng = random.Random(31)
        self.points = self._generate_points()

    @property
    def preset(self) -> AnomalyPreset:
        """Return the active anomaly preset."""
        return PRESETS[self.preset_index]

    @property
    def threshold(self) -> float:
        """Return the active anomaly threshold."""
        return THRESHOLDS[self.threshold_index]

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
        """Draw the anomaly detection lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_plot_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self.threshold_index = DEFAULT_THRESHOLD_INDEX
            self.points = self._generate_points()
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.threshold_index = max(0, self.threshold_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.threshold_index = min(len(THRESHOLDS) - 1, self.threshold_index + 1)
        elif key == pygame.K_s:
            self.show_scores = not self.show_scores
        elif key in {pygame.K_0, pygame.K_KP0}:
            self.threshold_index = DEFAULT_THRESHOLD_INDEX
        elif key == pygame.K_r:
            self.preset_index = 0
            self.threshold_index = DEFAULT_THRESHOLD_INDEX
            self.show_scores = True
            self.points = self._generate_points()

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Anomaly Detection Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Move the threshold and watch alert noise trade off with missed anomalies.",
                "Przesuwaj threshold i obserwuj kompromis między szumem alertów a brakami.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_plot_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Score map", "Mapa score"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._threshold_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        plot_rect = pygame.Rect(rect.x + 58, rect.y + 96, 590, 280)
        self._draw_scatter(surface, plot_rect)
        self._draw_score_strip(surface, pygame.Rect(rect.x + 58, rect.y + 408, 590, 42))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 34),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        counts = self.confusion_counts()
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("threshold", "threshold"), self._threshold_label()),
            (self._label("alerts", "alerty"), str(counts.true_positive + counts.false_positive)),
            (self._label("precision", "precision"), self._precision_label()),
            (self._label("recall", "recall"), self._recall_label()),
            (self._label("false positives", "false positives"), str(counts.false_positive)),
            (self._label("missed anomalies", "pominięte anomalie"), str(counts.false_negative)),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Alert readout", "Odczyt alertów"),
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
                "1-3: dataset | -/=: threshold | 0: default | S: scores | R: reset",
                "1-3: dataset | -/=: threshold | 0: default | S: score | R: reset",
            ),
            (58, 642),
            self._font_small,
            TEXT,
        )

    def _draw_scatter(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

        threshold_radius = round(self.threshold * min(rect.width, rect.height) / 2)
        center = self._to_screen(rect, Point(*self.preset.normal_center, False))
        pygame.draw.circle(surface, SECONDARY, center, threshold_radius, width=2)

        for point in self.points:
            score = self.score_for_point(point)
            flagged = score >= self.threshold
            color = self._point_color(point, flagged)
            radius = 6 if point.true_anomaly else 4
            pygame.draw.circle(surface, color, self._to_screen(rect, point), radius)
            if self.show_scores and flagged:
                pygame.draw.circle(surface, TEXT, self._to_screen(rect, point), radius + 3, width=1)

    def _draw_score_strip(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        threshold_x = rect.x + round(self.threshold * rect.width)
        pygame.draw.line(surface, SECONDARY, (threshold_x, rect.top), (threshold_x, rect.bottom), 2)
        for point in self.points:
            x = rect.x + round(self.score_for_point(point) * rect.width)
            y = rect.centery + (-7 if point.true_anomaly else 7)
            color = WARNING if point.true_anomaly else ACCENT
            pygame.draw.circle(surface, color, (x, y), 3)
        self._draw_text(surface, "score", (rect.x, rect.y - 22), self._font_small, MUTED_TEXT)

    def _generate_points(self) -> list[Point]:
        self._rng.seed(31 + self.preset_index * 1000)
        points: list[Point] = []
        center_x, center_y = self.preset.normal_center
        spread_x, spread_y = self.preset.normal_spread
        for _ in range(self.preset.normal_count):
            points.append(
                Point(
                    self._clamp(center_x + self._rng.gauss(0.0, spread_x)),
                    self._clamp(center_y + self._rng.gauss(0.0, spread_y)),
                    False,
                )
            )
        for index in range(self.preset.anomaly_count):
            anomaly_center = self.preset.anomaly_centers[index % len(self.preset.anomaly_centers)]
            points.append(
                Point(
                    self._clamp(anomaly_center[0] + self._rng.gauss(0.0, 0.07)),
                    self._clamp(anomaly_center[1] + self._rng.gauss(0.0, 0.07)),
                    True,
                )
            )
        return points

    def score_for_point(self, point: Point) -> float:
        """Return an anomaly score based on normalized distance from normal center."""
        center_x, center_y = self.preset.normal_center
        spread_x, spread_y = self.preset.normal_spread
        distance = math.sqrt(
            ((point.x - center_x) / (spread_x * 3.0)) ** 2
            + ((point.y - center_y) / (spread_y * 3.0)) ** 2
        )
        return max(0.0, min(1.0, distance))

    def confusion_counts(self) -> ConfusionCounts:
        """Return confusion counts for the active anomaly threshold."""
        true_positive = false_positive = false_negative = true_negative = 0
        for point in self.points:
            flagged = self.score_for_point(point) >= self.threshold
            if point.true_anomaly and flagged:
                true_positive += 1
            elif not point.true_anomaly and flagged:
                false_positive += 1
            elif point.true_anomaly:
                false_negative += 1
            else:
                true_negative += 1
        return ConfusionCounts(true_positive, false_positive, false_negative, true_negative)

    def _precision(self) -> float:
        counts = self.confusion_counts()
        alerts = counts.true_positive + counts.false_positive
        if alerts == 0:
            return 0.0
        return counts.true_positive / alerts

    def _recall(self) -> float:
        counts = self.confusion_counts()
        positives = counts.true_positive + counts.false_negative
        if positives == 0:
            return 0.0
        return counts.true_positive / positives

    def _threshold_label(self) -> str:
        return f"{self.threshold:.0%} ({self.threshold_index + 1}/3)"

    def _precision_label(self) -> str:
        return f"{self._precision():.0%}"

    def _recall_label(self) -> str:
        return f"{self._recall():.0%}"

    def _diagnosis_key(self) -> str:
        counts = self.confusion_counts()
        if counts.false_negative > counts.true_positive:
            return "misses"
        if counts.false_positive > counts.true_positive:
            return "noise"
        return "balanced"

    def _diagnosis_label(self) -> str:
        key = self._diagnosis_key()
        if key == "misses":
            return self._label("missed anomalies", "pominięte anomalie")
        if key == "noise":
            return self._label("alert noise", "szum alertów")
        return self._label("usable threshold", "użyteczny threshold")

    def _active_takeaway(self) -> str:
        key = self._diagnosis_key()
        if key == "misses":
            return self._label(
                "The threshold is strict: fewer alerts, but more missed anomalies.",
                "Threshold jest surowy: mniej alertów, ale więcej pominiętych anomalii.",
            )
        if key == "noise":
            return self._label(
                "The threshold is loose: recall improves, but alert noise grows.",
                "Threshold jest luźny: recall rośnie, ale rośnie też szum alertów.",
            )
        return self._label(
            "This threshold balances alert volume with anomaly coverage.",
            "Ten threshold równoważy liczbę alertów i pokrycie anomalii.",
        )

    def _point_color(self, point: Point, flagged: bool) -> tuple[int, int, int]:
        if point.true_anomaly and flagged:
            return WARNING
        if point.true_anomaly:
            return SECONDARY
        if flagged:
            return MUTED_TEXT
        return ACCENT

    def _to_screen(self, rect: pygame.Rect, point: Point) -> tuple[int, int]:
        x = rect.x + round((point.x + 1.0) / 2.0 * rect.width)
        y = rect.bottom - round((point.y + 1.0) / 2.0 * rect.height)
        return (x, y)

    def _clamp(self, value: float) -> float:
        return max(-0.94, min(0.94, value))

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)

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

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl
        return en


def create_anomaly_detection_lab_scene(context: AppContext) -> AnomalyDetectionLabScene:
    """Create the unified shell Anomaly Detection Lab scene."""
    return AnomalyDetectionLabScene(context)
