"""Native Model Monitoring Drift Lab skeleton scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BACKGROUND: Final[tuple[int, int, int]] = (19, 23, 28)
PANEL: Final[tuple[int, int, int]] = (34, 40, 47)
PLOT_BG: Final[tuple[int, int, int]] = (15, 18, 23)
GRID: Final[tuple[int, int, int]] = (51, 58, 67)
TEXT: Final[tuple[int, int, int]] = (236, 239, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (164, 173, 184)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
SECONDARY: Final[tuple[int, int, int]] = (248, 183, 96)
GOOD: Final[tuple[int, int, int]] = (147, 218, 155)
WARNING: Final[tuple[int, int, int]] = (246, 132, 134)

WINDOW_SIZE: Final[int] = 8
DEFAULT_THRESHOLD_INDEX: Final[int] = 2
THRESHOLDS: Final[tuple[float, ...]] = (0.10, 0.15, 0.20, 0.25, 0.30)
SIGNALS: Final[tuple[str, ...]] = ("data drift", "metric drift")


@dataclass(frozen=True, slots=True)
class MonitoringPreset:
    """Static monitoring time-series preset."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    data_drift: tuple[float, ...]
    metric_drift: tuple[float, ...]

    def name_for_language(self, language: str) -> str:
        """Return localized preset name."""
        if language == "pl":
            return self.name_pl
        return self.name_en

    def summary_for_language(self, language: str) -> str:
        """Return localized preset summary."""
        if language == "pl":
            return self.summary_pl
        return self.summary_en


PRESETS: Final[tuple[MonitoringPreset, ...]] = (
    MonitoringPreset(
        name_en="Stable baseline",
        name_pl="Stable baseline",
        summary_en="Small fluctuations stay close to the baseline window.",
        summary_pl="Małe wahania zostają blisko baseline window.",
        data_drift=(0.04, 0.06, 0.05, 0.07, 0.06, 0.05, 0.08, 0.06, 0.07, 0.08, 0.06, 0.07),
        metric_drift=(0.05, 0.04, 0.06, 0.05, 0.06, 0.07, 0.06, 0.08, 0.07, 0.06, 0.08, 0.07),
    ),
    MonitoringPreset(
        name_en="Data drift first",
        name_pl="Data drift first",
        summary_en="Inputs start moving before model quality clearly reacts.",
        summary_pl="Wejścia zaczynają się zmieniać, zanim jakość modelu wyraźnie reaguje.",
        data_drift=(0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.22, 0.31, 0.40, 0.48, 0.55, 0.60),
        metric_drift=(0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.10, 0.13, 0.16, 0.20, 0.24, 0.28),
    ),
    MonitoringPreset(
        name_en="Metric drift first",
        name_pl="Metric drift first",
        summary_en="Quality degrades even though input drift looks modest.",
        summary_pl="Jakość spada, mimo że input drift wygląda umiarkowanie.",
        data_drift=(0.05, 0.06, 0.06, 0.07, 0.08, 0.10, 0.11, 0.13, 0.14, 0.16, 0.17, 0.18),
        metric_drift=(0.05, 0.06, 0.08, 0.10, 0.14, 0.19, 0.24, 0.30, 0.35, 0.39, 0.42, 0.46),
    ),
)


class ModelMonitoringDriftScene:
    """First native visual slice for monitoring and drift."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic monitoring preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.signal_index = 0
        self.threshold_index = DEFAULT_THRESHOLD_INDEX
        self.investigation_acknowledged = False

    @property
    def preset(self) -> MonitoringPreset:
        """Return the active monitoring preset."""
        return PRESETS[self.preset_index]

    @property
    def signal(self) -> str:
        """Return the active monitored signal."""
        return SIGNALS[self.signal_index]

    @property
    def threshold(self) -> float:
        """Return the active alert threshold."""
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
        """Draw the monitoring preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_signal_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self.investigation_acknowledged = False
        elif key in {pygame.K_m, pygame.K_d}:
            self.signal_index = (self.signal_index + 1) % len(SIGNALS)
            self.investigation_acknowledged = False
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.threshold_index = max(0, self.threshold_index - 1)
            self.investigation_acknowledged = False
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.threshold_index = min(len(THRESHOLDS) - 1, self.threshold_index + 1)
            self.investigation_acknowledged = False
        elif key == pygame.K_a:
            self.investigation_acknowledged = (
                self._alert_active() and not self.investigation_acknowledged
            )
        elif key == pygame.K_r:
            self.preset_index = 0
            self.signal_index = 0
            self.threshold_index = DEFAULT_THRESHOLD_INDEX
            self.investigation_acknowledged = False

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Model Monitoring Drift Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Compare baseline and current windows before trusting an alert.",
                "Porównaj baseline i current window, zanim zaufasz alertowi.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_signal_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Monitoring timeline", "Monitoring timeline"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        plot_rect = self._timeline_plot_rect(rect)
        self._draw_plot_frame(surface, plot_rect)
        self._draw_window_band(surface, plot_rect, 0, WINDOW_SIZE, GOOD)
        self._draw_window_band(
            surface,
            plot_rect,
            len(self._active_series()) - WINDOW_SIZE,
            len(self._active_series()),
            WARNING,
        )
        self._draw_threshold(surface, plot_rect)
        self._draw_series(surface, plot_rect, self._active_series())
        self._draw_text(
            surface,
            self._label("baseline", "baseline"),
            (plot_rect.x, plot_rect.y - 24),
            self._font_small,
            GOOD,
        )
        self._draw_text(
            surface,
            self._label("current window", "current window"),
            (plot_rect.right - 118, plot_rect.y - 24),
            self._font_small,
            WARNING,
        )
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 64),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Controls and signals", "Sterowanie i sygnały"),
            (rect.x + 22, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        rows = (
            (self._label("signal", "sygnał"), self.signal),
            (self._label("threshold", "threshold"), f"{self.threshold:.0%}"),
            (self._label("first alert", "pierwszy alert"), self._first_alert_label()),
            (self._label("lead signal", "lead signal"), self._lead_signal_label()),
            (self._label("baseline mean", "baseline mean"), f"{self._window_mean(0):.0%}"),
            (self._label("current mean", "current mean"), f"{self._window_mean(-1):.0%}"),
            (self._label("severity", "severity"), self._severity_label()),
            (self._label("investigation", "analiza"), self._investigation_state_label()),
        )
        y = rect.y + 68
        for label, value in rows:
            self._draw_text(surface, f"{label}: {value}", (rect.x + 22, y), self._font_small, TEXT)
            y += 28
        y += 12
        for index, preset in enumerate(PRESETS):
            color = ACCENT if index == self.preset_index else MUTED_TEXT
            self._draw_text(
                surface,
                f"{index + 1}. {preset.name_for_language(self._language)}",
                (rect.x + 22, y),
                self._font_small,
                color,
            )
            y += 30
        self._draw_wrapped(
            surface,
            self._active_takeaway(),
            (rect.x + 22, rect.bottom - 96),
            rect.width - 44,
            self._font_small,
            SECONDARY,
            line_height=17,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: decide whether a change deserves investigation, not automatic panic.",
                "Cel: zdecyduj, czy zmiana wymaga analizy, a nie automatycznej paniki.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _active_series(self) -> tuple[float, ...]:
        if self.signal_index == 0:
            return self.preset.data_drift
        return self.preset.metric_drift

    def _window_mean(self, window: int) -> float:
        series = self._active_series()
        values = series[:WINDOW_SIZE] if window == 0 else series[-WINDOW_SIZE:]
        return sum(values) / len(values)

    def _drift_gap(self) -> float:
        return max(0.0, self._window_mean(-1) - self._window_mean(0))

    def _alert_active(self) -> bool:
        return self._drift_gap() >= self.threshold

    def _first_alert_index(self) -> int | None:
        return self._first_alert_index_for(self._active_series())

    def _first_alert_index_for(self, series: tuple[float, ...]) -> int | None:
        for index, value in enumerate(series):
            if value >= self.threshold:
                return index
        return None

    def _first_alert_label(self) -> str:
        first_alert_index = self._first_alert_index()
        if first_alert_index is None:
            return self._label("none", "brak")
        return f"t{first_alert_index}"

    def _lead_signal_key(self) -> str:
        data_index = self._first_alert_index_for(self.preset.data_drift)
        metric_index = self._first_alert_index_for(self.preset.metric_drift)
        if data_index is None and metric_index is None:
            return "none"
        if data_index == metric_index:
            return "tie"
        if metric_index is None or (data_index is not None and data_index < metric_index):
            return "data drift"
        return "metric drift"

    def _lead_signal_label(self) -> str:
        lead_signal = self._lead_signal_key()
        if lead_signal == "none":
            return self._label("none", "brak")
        if lead_signal == "tie":
            return self._label("tie", "remis")
        return lead_signal

    def _severity_label(self) -> str:
        severity = self._severity_key()
        if severity == "high":
            return self._label("high", "wysoka")
        if severity == "watch":
            return self._label("watch", "obserwuj")
        return self._label("normal", "normal")

    def _severity_key(self) -> str:
        if not self._alert_active():
            return "normal"
        if self._drift_gap() >= self.threshold * 1.5:
            return "high"
        return "watch"

    def _investigation_state_label(self) -> str:
        if self.investigation_acknowledged:
            return self._label("acknowledged", "potwierdzona")
        if self._alert_active():
            return self._label("needs review", "do sprawdzenia")
        return self._label("not needed", "niepotrzebna")

    def _recommendation_label(self) -> str:
        if self.investigation_acknowledged:
            return self._label("document finding", "opisz wniosek")
        if self._severity_key() == "high":
            return self._label("investigate now", "sprawdź teraz")
        if self._alert_active():
            return self._label("compare signals", "porównaj sygnały")
        return self._label("watch", "obserwuj")

    def _active_takeaway(self) -> str:
        recommendation = self._recommendation_label()
        if self.investigation_acknowledged:
            return self._label(
                f"Next: {recommendation}. Compare data, metric, and business context.",
                f"Dalej: {recommendation}. Porównaj data, metric i kontekst biznesowy.",
            )
        if self._severity_key() == "high":
            return self._label(
                f"Next: {recommendation}. Current window is far from baseline.",
                f"Dalej: {recommendation}. Current window jest daleko od baseline.",
            )
        if self._alert_active():
            return self._label(
                f"Next: {recommendation}. Press A when ready to investigate.",
                f"Dalej: {recommendation}. Naciśnij A, gdy chcesz przejść do analizy.",
            )
        return self._label(
            f"Next: {recommendation}. Keep watching whether the gap becomes persistent.",
            f"Dalej: {recommendation}. Obserwuj, czy luka staje się trwała.",
        )

    def _timeline_plot_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Return the main time-series plot rectangle."""
        return pygame.Rect(rect.x + 52, rect.y + 92, rect.width - 104, rect.height - 188)

    def _draw_series(
        self, surface: pygame.Surface, rect: pygame.Rect, series: tuple[float, ...]
    ) -> None:
        points = [self._series_position(rect, index, value) for index, value in enumerate(series)]
        if len(points) >= 2:
            pygame.draw.lines(surface, ACCENT, False, points, 3)
        first_alert_index = self._first_alert_index()
        for index, point in enumerate(points):
            color = WARNING if series[index] >= self.threshold else ACCENT
            pygame.draw.circle(surface, color, point, 5)
            if index == first_alert_index:
                pygame.draw.circle(surface, WARNING, point, 11, width=2)
                self._draw_text(
                    surface,
                    self._label("first alert", "pierwszy alert"),
                    (min(point[0] + 10, rect.right - 108), max(rect.top + 8, point[1] - 32)),
                    self._font_small,
                    WARNING,
                )

    def _draw_threshold(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        y = self._series_position(rect, 0, self.threshold)[1]
        pygame.draw.line(surface, SECONDARY, (rect.left, y), (rect.right, y), 2)
        self._draw_text(surface, "threshold", (rect.x + 10, y - 22), self._font_small, SECONDARY)

    def _draw_window_band(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        start_index: int,
        end_index: int,
        color: tuple[int, int, int],
    ) -> None:
        left = self._series_position(rect, start_index, 0)[0]
        right = self._series_position(rect, end_index - 1, 0)[0]
        band_rect = pygame.Rect(left - 8, rect.y + 8, right - left + 16, rect.height - 16)
        overlay = pygame.Surface(band_rect.size, pygame.SRCALPHA)
        overlay.fill((*color, 24))
        surface.blit(overlay, band_rect.topleft)

    def _series_position(self, rect: pygame.Rect, index: int, value: float) -> tuple[int, int]:
        series = self._active_series()
        x = rect.x + round(index / (len(series) - 1) * rect.width)
        y = rect.bottom - round(max(0.0, min(0.5, value)) / 0.5 * rect.height)
        return (x, y)

    def _draw_plot_frame(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

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


def create_model_monitoring_drift_scene(context: AppContext) -> ModelMonitoringDriftScene:
    """Create the unified shell Model Monitoring Drift Lab scene."""
    return ModelMonitoringDriftScene(context)
