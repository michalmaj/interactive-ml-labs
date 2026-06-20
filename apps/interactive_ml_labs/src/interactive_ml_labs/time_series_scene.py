"""Native Time Series Forecasting Lab scene for the unified shell."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
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
BAND: Final[tuple[int, int, int, int]] = (118, 205, 247, 36)

HORIZONS: Final[tuple[int, ...]] = (4, 6, 8)
SEASON_LENGTH: Final[int] = 6


class ForecastModel(StrEnum):
    """Forecasting model shown in the lab."""

    NAIVE = "naive"
    MOVING_AVERAGE = "moving_average"
    TREND_SEASONAL = "trend_seasonal"


@dataclass(frozen=True, slots=True)
class TimeSeriesPreset:
    """Static time-series scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    values: tuple[float, ...]

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


def _make_series(
    *,
    base: float,
    trend: float,
    amplitude: float,
    drift_after: int | None = None,
    drift: float = 0.0,
) -> tuple[float, ...]:
    """Create a deterministic teaching series."""
    values: list[float] = []
    for index in range(36):
        seasonal = amplitude * math.sin((index % SEASON_LENGTH) / SEASON_LENGTH * math.tau)
        local_trend = trend * index
        drift_value = 0.0
        if drift_after is not None and index >= drift_after:
            drift_value = drift * (index - drift_after + 1)
        ripple = 0.18 * math.sin(index * 1.7)
        values.append(base + local_trend + seasonal + drift_value + ripple)
    return tuple(values)


PRESETS: Final[tuple[TimeSeriesPreset, ...]] = (
    TimeSeriesPreset(
        name_en="Seasonal demand",
        name_pl="Sezonowy popyt",
        summary_en=(
            "A regular seasonal pattern rewards models that remember the cycle, "
            "not only the last value."
        ),
        summary_pl=(
            "Regularny sezonowy wzorzec premiuje modele, które pamiętają cykl, "
            "a nie tylko ostatnią wartość."
        ),
        values=_make_series(base=9.0, trend=0.08, amplitude=1.25),
    ),
    TimeSeriesPreset(
        name_en="Trend shift",
        name_pl="Zmiana trendu",
        summary_en=(
            "The final segment grows faster, so a forecast fitted on old history "
            "starts lagging behind."
        ),
        summary_pl=(
            "Końcowy fragment rośnie szybciej, więc forecast dopasowany do starej "
            "historii zaczyna zostawać w tyle."
        ),
        values=_make_series(base=7.4, trend=0.04, amplitude=0.7, drift_after=25, drift=0.33),
    ),
    TimeSeriesPreset(
        name_en="Noisy operations",
        name_pl="Szum operacyjny",
        summary_en=(
            "Noise hides part of the pattern. A smoother forecast may be calmer, "
            "but it can miss sharp turns."
        ),
        summary_pl=(
            "Szum ukrywa część wzorca. Gładszy forecast jest spokojniejszy, "
            "ale może przegapić ostre zwroty."
        ),
        values=tuple(
            value + 0.55 * math.sin(index * 2.9)
            for index, value in enumerate(_make_series(base=8.2, trend=0.05, amplitude=0.95))
        ),
    ),
)


class TimeSeriesForecastingLabScene:
    """Interactive forecasting lab for Level 3."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic forecasting scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.model = ForecastModel.TREND_SEASONAL
        self.horizon_index = 1
        self.show_uncertainty = True
        self.show_residuals = True

    @property
    def preset(self) -> TimeSeriesPreset:
        """Return the active time-series preset."""
        return PRESETS[self.preset_index]

    @property
    def horizon(self) -> int:
        """Return the active forecast horizon."""
        return HORIZONS[self.horizon_index]

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
        """Draw the forecasting lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_chart_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key == pygame.K_m:
            self.model = self._next_model()
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.horizon_index = max(0, self.horizon_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.horizon_index = min(len(HORIZONS) - 1, self.horizon_index + 1)
        elif key == pygame.K_u:
            self.show_uncertainty = not self.show_uncertainty
        elif key == pygame.K_e:
            self.show_residuals = not self.show_residuals
        elif key == pygame.K_r:
            self.preset_index = 0
            self.model = ForecastModel.TREND_SEASONAL
            self.horizon_index = 1
            self.show_uncertainty = True
            self.show_residuals = True

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Time Series Forecasting Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Forecast the holdout window and compare trend, seasonality, and residuals.",
                "Prognozuj holdout window i porównuj trend, sezonowość oraz residuals.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_chart_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("History, forecast, holdout", "Historia, forecast, holdout"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._model_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        plot_rect = pygame.Rect(rect.x + 58, rect.y + 94, 590, 274)
        self._draw_forecast_plot(surface, plot_rect)
        if self.show_residuals:
            self._draw_residual_strip(surface, pygame.Rect(rect.x + 58, rect.y + 400, 590, 48))
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
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("model", "model"), self._model_label()),
            (self._label("horizon", "horizon"), str(self.horizon)),
            ("MAE", f"{self.mae():.2f}"),
            ("RMSE", f"{self.rmse():.2f}"),
            (self._label("bias", "bias"), f"{self.bias():+.2f}"),
            (self._label("uncertainty", "uncertainty"), self._uncertainty_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Forecast readout", "Odczyt forecastu"),
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
                "1-3: data | M: model | -/=: horizon | U: uncertainty | E: residuals | R: reset",
                "1-3: dane | M: model | -/=: horizon | U: uncertainty | E: residuals | R: reset",
            ),
            (58, 642),
            self._font_small,
            TEXT,
        )

    def _draw_forecast_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_grid(surface, rect)

        values = self.preset.values
        split_index = len(values) - self.horizon
        forecast = self.forecast_values()
        all_values = (*values, *forecast)
        value_min = min(all_values) - 0.7
        value_max = max(all_values) + 0.7

        split_x = self._to_screen(
            rect, split_index - 1, values[split_index - 1], value_min, value_max
        )[0]
        pygame.draw.line(surface, MUTED_TEXT, (split_x, rect.top), (split_x, rect.bottom), 1)
        self._draw_text(
            surface, "holdout", (split_x + 8, rect.y + 10), self._font_small, MUTED_TEXT
        )

        history_points = [
            self._to_screen(rect, index, value, value_min, value_max)
            for index, value in enumerate(values[:split_index])
        ]
        holdout_points = [
            self._to_screen(rect, index, value, value_min, value_max)
            for index, value in enumerate(values[split_index:], start=split_index)
        ]
        forecast_points = [
            self._to_screen(rect, index, value, value_min, value_max)
            for index, value in enumerate(forecast, start=split_index)
        ]

        if self.show_uncertainty:
            self._draw_uncertainty_band(
                surface,
                rect,
                forecast,
                split_index,
                value_min,
                value_max,
            )

        pygame.draw.lines(surface, ACCENT, False, history_points, 3)
        pygame.draw.lines(surface, SECONDARY, False, holdout_points, 3)
        pygame.draw.lines(surface, WARNING, False, forecast_points, 3)
        for point in history_points[::3]:
            pygame.draw.circle(surface, ACCENT, point, 3)
        for point in holdout_points:
            pygame.draw.circle(surface, SECONDARY, point, 4)
        for point in forecast_points:
            pygame.draw.circle(surface, WARNING, point, 4)

        self._draw_legend(surface, rect)

    def _draw_uncertainty_band(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        forecast: tuple[float, ...],
        split_index: int,
        value_min: float,
        value_max: float,
    ) -> None:
        band = self.uncertainty_width()
        upper = [
            self._to_screen(rect, index, value + band, value_min, value_max)
            for index, value in enumerate(forecast, start=split_index)
        ]
        lower = [
            self._to_screen(rect, index, value - band, value_min, value_max)
            for index, value in enumerate(forecast, start=split_index)
        ]
        band_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
        shifted = [(x - rect.x, y - rect.y) for x, y in (*upper, *reversed(lower))]
        pygame.draw.polygon(band_surface, BAND, shifted)
        surface.blit(band_surface, rect.topleft)

    def _draw_residual_strip(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        residuals = self.residuals()
        if not residuals:
            return

        max_abs = max(0.1, *(abs(value) for value in residuals))
        bar_width = max(10, (rect.width - 32) // len(residuals) - 6)
        zero_y = rect.centery
        pygame.draw.line(surface, GRID, (rect.x + 12, zero_y), (rect.right - 12, zero_y), 1)
        for index, residual in enumerate(residuals):
            x = rect.x + 18 + index * (bar_width + 6)
            height = round((abs(residual) / max_abs) * (rect.height / 2 - 8))
            top = zero_y - height if residual >= 0 else zero_y
            color = GOOD if abs(residual) <= self.mae() else WARNING
            pygame.draw.rect(
                surface, color, pygame.Rect(x, top, bar_width, height), border_radius=3
            )

        self._draw_text(
            surface,
            self._label("residuals", "residuals"),
            (rect.x + 12, rect.y - 20),
            self._font_small,
            MUTED_TEXT,
        )

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_legend(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        items = (
            (self._label("history", "historia"), ACCENT),
            (self._label("actual", "actual"), SECONDARY),
            ("forecast", WARNING),
        )
        x = rect.x + 14
        y = rect.bottom - 24
        for label, color in items:
            pygame.draw.circle(surface, color, (x, y + 7), 5)
            self._draw_text(surface, label, (x + 12, y), self._font_small, TEXT)
            x += self._font_small.size(label)[0] + 58

    def forecast_values(self) -> tuple[float, ...]:
        """Return forecast values for the active holdout horizon."""
        values = self.preset.values
        split_index = len(values) - self.horizon
        train = values[:split_index]
        if self.model == ForecastModel.NAIVE:
            return tuple(train[-1] for _ in range(self.horizon))
        if self.model == ForecastModel.MOVING_AVERAGE:
            return self._moving_average_forecast(train)
        return self._trend_seasonal_forecast(train, split_index)

    def residuals(self) -> tuple[float, ...]:
        """Return actual minus forecast residuals over holdout."""
        values = self.preset.values
        actual = values[len(values) - self.horizon :]
        forecast = self.forecast_values()
        return tuple(
            actual_value - forecast_value
            for actual_value, forecast_value in zip(actual, forecast, strict=True)
        )

    def mae(self) -> float:
        """Return mean absolute error over holdout."""
        return sum(abs(value) for value in self.residuals()) / self.horizon

    def rmse(self) -> float:
        """Return root mean squared error over holdout."""
        return math.sqrt(sum(value * value for value in self.residuals()) / self.horizon)

    def bias(self) -> float:
        """Return average residual over holdout."""
        return sum(self.residuals()) / self.horizon

    def uncertainty_width(self) -> float:
        """Return a simple uncertainty width from recent residual scale."""
        values = self.preset.values
        split_index = len(values) - self.horizon
        train = values[:split_index]
        fitted = self._one_step_backtest(train)
        if not fitted:
            return 0.7
        errors = [abs(actual - prediction) for actual, prediction in fitted]
        return max(0.45, sum(errors[-10:]) / min(10, len(errors)) * 1.35)

    def _moving_average_forecast(self, train: tuple[float, ...]) -> tuple[float, ...]:
        history = list(train)
        forecast: list[float] = []
        window = SEASON_LENGTH
        for _ in range(self.horizon):
            value = sum(history[-window:]) / min(window, len(history))
            forecast.append(value)
            history.append(value)
        return tuple(forecast)

    def _trend_seasonal_forecast(
        self,
        train: tuple[float, ...],
        split_index: int,
    ) -> tuple[float, ...]:
        slope = (train[-1] - train[-SEASON_LENGTH - 1]) / SEASON_LENGTH
        recent_average = sum(train[-SEASON_LENGTH:]) / SEASON_LENGTH
        seasonal_offsets = tuple(value - recent_average for value in train[-SEASON_LENGTH:])
        forecast = []
        for step in range(self.horizon):
            seasonal = seasonal_offsets[(split_index + step) % SEASON_LENGTH]
            forecast.append(train[-1] + slope * (step + 1) + seasonal)
        return tuple(forecast)

    def _one_step_backtest(self, train: tuple[float, ...]) -> list[tuple[float, float]]:
        fitted: list[tuple[float, float]] = []
        for index in range(SEASON_LENGTH + 3, len(train)):
            history = train[:index]
            if self.model == ForecastModel.NAIVE:
                prediction = history[-1]
            elif self.model == ForecastModel.MOVING_AVERAGE:
                prediction = sum(history[-SEASON_LENGTH:]) / SEASON_LENGTH
            else:
                prediction = self._trend_seasonal_forecast(history, index)[0]
            fitted.append((train[index], prediction))
        return fitted

    def _next_model(self) -> ForecastModel:
        models = tuple(ForecastModel)
        index = models.index(self.model)
        return models[(index + 1) % len(models)]

    def _model_label(self) -> str:
        if self.model == ForecastModel.NAIVE:
            return self._label("naive last value", "naive last value")
        if self.model == ForecastModel.MOVING_AVERAGE:
            return self._label("moving average", "moving average")
        return self._label("trend + seasonality", "trend + sezonowość")

    def _uncertainty_label(self) -> str:
        state = (
            self._label("shown", "widoczna")
            if self.show_uncertainty
            else self._label("hidden", "ukryta")
        )
        return f"{state} +/- {self.uncertainty_width():.2f}"

    def _diagnosis_label(self) -> str:
        bias = self.bias()
        if abs(bias) > self.uncertainty_width() * 0.55:
            return self._label("biased forecast", "forecast z bias")
        if self.rmse() > self.uncertainty_width() * 1.15:
            return self._label("wide residuals", "szerokie residuals")
        return self._label("usable forecast", "użyteczny forecast")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_label()
        if "bias" in diagnosis or "biased" in diagnosis:
            return self._label(
                "Forecast bias means the model is systematically late or early, not just noisy.",
                (
                    "Bias forecastu oznacza systematyczne spóźnienie albo "
                    "wyprzedzenie, nie tylko noise."
                ),
            )
        if "residual" in diagnosis:
            return self._label(
                "Wide residuals say the future is less predictable than the line suggests.",
                (
                    "Szerokie residuals mówią, że przyszłość jest mniej "
                    "przewidywalna niż sugeruje linia."
                ),
            )
        return self._label(
            "Good forecasts are judged on holdout error, not on how smooth the line looks.",
            "Dobry forecast oceniamy po holdout error, a nie po tym, jak gładko wygląda linia.",
        )

    def _to_screen(
        self,
        rect: pygame.Rect,
        index: int,
        value: float,
        value_min: float,
        value_max: float,
    ) -> tuple[int, int]:
        max_index = len(self.preset.values) - 1
        x = rect.x + round((index / max_index) * rect.width)
        ratio = (value - value_min) / max(0.1, value_max - value_min)
        y = rect.bottom - round(ratio * rect.height)
        return (x, y)

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
    ) -> int:
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
            y += line_height
        return y

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl
        return en


def create_time_series_forecasting_lab_scene(
    context: AppContext,
) -> TimeSeriesForecastingLabScene:
    """Create the unified shell Time Series Forecasting Lab scene."""
    return TimeSeriesForecastingLabScene(context)
