"""Native Calibration Lab skeleton scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BACKGROUND: Final[tuple[int, int, int]] = (20, 23, 27)
PANEL: Final[tuple[int, int, int]] = (34, 39, 45)
PLOT_BG: Final[tuple[int, int, int]] = (16, 19, 23)
GRID: Final[tuple[int, int, int]] = (48, 55, 63)
TEXT: Final[tuple[int, int, int]] = (236, 239, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (166, 174, 184)
ACCENT: Final[tuple[int, int, int]] = (118, 205, 247)
REFERENCE: Final[tuple[int, int, int]] = (128, 136, 148)
WARNING: Final[tuple[int, int, int]] = (244, 131, 133)
GOOD: Final[tuple[int, int, int]] = (151, 219, 156)
SECONDARY: Final[tuple[int, int, int]] = (247, 179, 101)
TEMPERATURE_VALUES: Final[tuple[float, ...]] = (0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
DEFAULT_TEMPERATURE_INDEX: Final[int] = 2
DECISION_THRESHOLD: Final[float] = 0.50
SIDE_ROW_STEP: Final[int] = 24
SIDE_NOTE_LINE_HEIGHT: Final[int] = 16


@dataclass(frozen=True, slots=True)
class CalibrationPreset:
    """Static probability preset for the calibration preview."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    samples: tuple[tuple[float, int], ...]

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


PRESETS: Final[tuple[CalibrationPreset, ...]] = (
    CalibrationPreset(
        name_en="Overconfident",
        name_pl="Overconfident",
        summary_en="Scores are too extreme for how often the model is right.",
        summary_pl="Score są zbyt skrajne względem tego, jak często model ma rację.",
        samples=(
            (0.05, 0),
            (0.08, 1),
            (0.14, 0),
            (0.18, 0),
            (0.24, 1),
            (0.32, 0),
            (0.41, 0),
            (0.52, 1),
            (0.61, 0),
            (0.70, 1),
            (0.78, 1),
            (0.86, 0),
            (0.92, 1),
            (0.96, 1),
            (0.98, 0),
        ),
    ),
    CalibrationPreset(
        name_en="Underconfident",
        name_pl="Underconfident",
        summary_en="Scores stay near the middle even when outcomes are easier.",
        summary_pl="Score zostają blisko środka, nawet gdy przypadki są łatwiejsze.",
        samples=(
            (0.28, 0),
            (0.31, 0),
            (0.34, 0),
            (0.38, 1),
            (0.42, 0),
            (0.46, 0),
            (0.49, 1),
            (0.52, 0),
            (0.55, 1),
            (0.58, 1),
            (0.61, 1),
            (0.64, 0),
            (0.68, 1),
            (0.71, 1),
            (0.74, 1),
        ),
    ),
    CalibrationPreset(
        name_en="Better calibrated",
        name_pl="Lepiej skalibrowane",
        summary_en="Predicted probabilities roughly match observed frequencies.",
        summary_pl="Predicted probabilities mniej więcej pasują do częstości obserwacji.",
        samples=(
            (0.06, 0),
            (0.12, 0),
            (0.18, 0),
            (0.24, 0),
            (0.30, 1),
            (0.38, 0),
            (0.46, 1),
            (0.54, 0),
            (0.62, 1),
            (0.68, 1),
            (0.74, 1),
            (0.80, 0),
            (0.86, 1),
            (0.92, 1),
            (0.97, 1),
        ),
    ),
)


class CalibrationLabScene:
    """First interactive slice for probability calibration."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic calibration preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.temperature_index = DEFAULT_TEMPERATURE_INDEX
        self.show_error_bars = True
        self.show_raw_scores = True

    @property
    def preset(self) -> CalibrationPreset:
        """Return the active calibration preset."""
        return PRESETS[self.preset_index]

    @property
    def temperature(self) -> float:
        """Return the active post-hoc temperature value."""
        return TEMPERATURE_VALUES[self.temperature_index]

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
        """Draw the calibration preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_reliability_diagram(surface, pygame.Rect(58, 132, 520, 474))
        self._draw_score_distribution(surface, pygame.Rect(620, 132, 300, 474))
        self._draw_side_panel(surface, pygame.Rect(960, 132, 260, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.temperature_index = max(0, self.temperature_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.temperature_index = min(len(TEMPERATURE_VALUES) - 1, self.temperature_index + 1)
        elif key == pygame.K_e:
            self.show_error_bars = not self.show_error_bars
        elif key == pygame.K_o:
            self.show_raw_scores = not self.show_raw_scores
        elif key == pygame.K_r:
            self.preset_index = 0
            self.temperature_index = DEFAULT_TEMPERATURE_INDEX
            self.show_error_bars = True
            self.show_raw_scores = True

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Calibration Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Check whether confidence scores match observed outcomes.",
                "Sprawdź, czy confidence scores pasują do obserwowanych wyników.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_reliability_diagram(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Reliability diagram", "Reliability diagram"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        plot_rect = pygame.Rect(rect.x + 48, rect.y + 84, rect.width - 96, rect.height - 146)
        self._draw_calibration_grid(surface, plot_rect)
        pygame.draw.line(surface, REFERENCE, plot_rect.bottomleft, plot_rect.topright, 2)

        bins = self._calibration_bins()
        bar_width = max(22, round(plot_rect.width / len(bins)) - 12)
        for index, calibration_bin in enumerate(bins):
            center_x = plot_rect.left + round((index + 0.5) * plot_rect.width / len(bins))
            expected_y = self._value_to_y(calibration_bin["confidence"], plot_rect)
            observed_y = self._value_to_y(calibration_bin["accuracy"], plot_rect)
            color = self._calibration_color(
                abs(calibration_bin["accuracy"] - calibration_bin["confidence"])
            )
            bar_rect = pygame.Rect(
                center_x - bar_width // 2,
                observed_y,
                bar_width,
                plot_rect.bottom - observed_y,
            )
            pygame.draw.rect(surface, color, bar_rect, border_radius=5)
            if self.show_error_bars:
                pygame.draw.line(
                    surface, WARNING, (center_x, expected_y), (center_x, observed_y), 3
                )
            self._draw_text(
                surface,
                str(calibration_bin["count"]),
                (center_x - 5, plot_rect.bottom + 14),
                self._font_small,
                MUTED_TEXT,
            )

        self._draw_text(
            surface,
            self._label("confidence", "confidence"),
            (plot_rect.centerx - 42, plot_rect.bottom + 42),
            self._font_small,
            MUTED_TEXT,
        )
        self._draw_text(
            surface,
            self._label("observed frequency", "częstość obserwacji"),
            (plot_rect.x - 24, plot_rect.y - 26),
            self._font_small,
            MUTED_TEXT,
        )

    def _draw_score_distribution(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Scores", "Score"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        plot_rect = self._score_distribution_plot_rect(rect)
        pygame.draw.rect(surface, PLOT_BG, plot_rect, border_radius=6)
        pygame.draw.rect(surface, GRID, plot_rect, width=1, border_radius=6)
        threshold_x = plot_rect.left + round(DECISION_THRESHOLD * plot_rect.width)
        pygame.draw.line(
            surface, REFERENCE, (threshold_x, plot_rect.top), (threshold_x, plot_rect.bottom), 2
        )
        self._draw_text(
            surface,
            self._label("threshold 0.5", "threshold 0.5"),
            (threshold_x - 42, plot_rect.top + 10),
            self._font_small,
            MUTED_TEXT,
        )
        if self.show_raw_scores:
            self._draw_score_points(
                surface,
                plot_rect,
                self.preset.samples,
                filled=False,
                y_offset=12,
            )
        self._draw_score_points(
            surface,
            plot_rect,
            self._active_samples(),
            filled=True,
            y_offset=0,
        )

        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, self._score_summary_top_y(rect)),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=18,
        )

    def _draw_score_points(
        self,
        surface: pygame.Surface,
        plot_rect: pygame.Rect,
        samples: tuple[tuple[float, int], ...],
        *,
        filled: bool,
        y_offset: int,
    ) -> None:
        for index, (probability, outcome) in enumerate(samples):
            x = plot_rect.left + round(probability * plot_rect.width)
            y = plot_rect.bottom - 20 - (index % 5) * 30 - y_offset
            color = GOOD if outcome == 1 else SECONDARY
            if filled:
                pygame.draw.circle(surface, color, (x, y), 7)
                pygame.draw.circle(surface, PLOT_BG, (x, y), 7, width=1)
                continue

            pygame.draw.circle(surface, color, (x, y), 9, width=2)

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Calibration", "Kalibracja"),
            (rect.x + 22, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        y = rect.y + 70
        for index, preset in enumerate(PRESETS):
            active = index == self.preset_index
            color = ACCENT if active else MUTED_TEXT
            self._draw_text(
                surface,
                f"{index + 1}. {preset.name_for_language(self._language)}",
                (rect.x + 22, y),
                self._font_body if active else self._font_small,
                color,
            )
            y += 38
        y += 16
        rows = (
            ("Brier", f"{self._brier_score():.3f}"),
            ("ECE", f"{self._expected_calibration_error():.3f}"),
            ("accuracy@0.5", f"{self._accuracy_at_threshold():.0%}"),
            ("temperature", f"{self.temperature:.2f}"),
            (self._label("samples", "próbki"), str(len(self.preset.samples))),
            (
                self._label("error bars", "error bars"),
                self._label("on", "wł.") if self.show_error_bars else self._label("off", "wył."),
            ),
            (
                self._label("raw scores", "raw score"),
                self._label("on", "wł.") if self.show_raw_scores else self._label("off", "wył."),
            ),
        )
        for label, value in rows:
            self._draw_text(surface, f"{label}: {value}", (rect.x + 22, y), self._font_small, TEXT)
            y += SIDE_ROW_STEP
        self._draw_wrapped(
            surface,
            self._label(
                "ECE summarizes the average gap between confidence and observed frequency.",
                "ECE streszcza średnią różnicę między confidence a obserwowaną częstością.",
            ),
            (rect.x + 22, self._side_note_top_y(rect)),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=SIDE_NOTE_LINE_HEIGHT,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: a model can be accurate and still poorly calibrated.",
                "Cel: model może mieć dobrą accuracy, a mimo to być źle skalibrowany.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _calibration_bins(self) -> tuple[dict[str, float | int], ...]:
        bins: list[dict[str, float | int]] = []
        for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
            upper = lower + 0.2
            samples = tuple(
                sample
                for sample in self._active_samples()
                if (lower <= sample[0] < upper) or (upper >= 1.0 and lower <= sample[0] <= upper)
            )
            if samples:
                confidence = sum(probability for probability, _outcome in samples) / len(samples)
                accuracy = sum(outcome for _probability, outcome in samples) / len(samples)
            else:
                confidence = lower + 0.1
                accuracy = 0.0
            bins.append({"confidence": confidence, "accuracy": accuracy, "count": len(samples)})
        return tuple(bins)

    def _brier_score(self) -> float:
        samples = self._active_samples()
        errors = ((probability - outcome) ** 2 for probability, outcome in samples)
        return sum(errors) / len(samples)

    def _expected_calibration_error(self) -> float:
        total = len(self._active_samples())
        return sum(
            (calibration_bin["count"] / total)
            * abs(calibration_bin["accuracy"] - calibration_bin["confidence"])
            for calibration_bin in self._calibration_bins()
        )

    def _accuracy_at_threshold(self) -> float:
        """Return classification accuracy when probabilities are thresholded at 0.5."""
        samples = self._active_samples()
        correct = sum(
            1
            for probability, outcome in samples
            if int(probability >= DECISION_THRESHOLD) == outcome
        )
        return correct / len(samples)

    def _active_samples(self) -> tuple[tuple[float, int], ...]:
        """Return samples after applying the current temperature scaling."""
        return tuple(
            (self._apply_temperature(probability), outcome)
            for probability, outcome in self.preset.samples
        )

    def _apply_temperature(self, probability: float) -> float:
        """Apply post-hoc temperature scaling to one probability score."""
        clipped = min(0.999, max(0.001, probability))
        logit = log(clipped / (1.0 - clipped))
        scaled = 1.0 / (1.0 + exp(-(logit / self.temperature)))
        return min(0.999, max(0.001, scaled))

    def _draw_calibration_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 5):
            x = rect.left + round(step * rect.width / 5)
            y = rect.top + round(step * rect.height / 5)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _value_to_y(self, value: float, rect: pygame.Rect) -> int:
        return rect.bottom - round(value * rect.height)

    def _calibration_color(self, gap: float) -> tuple[int, int, int]:
        if gap < 0.10:
            return GOOD
        if gap < 0.22:
            return SECONDARY
        return WARNING

    def _score_distribution_plot_rect(self, rect: pygame.Rect) -> pygame.Rect:
        """Return the chart area for score samples."""
        return pygame.Rect(rect.x + 34, rect.y + 86, rect.width - 68, rect.height - 196)

    def _score_summary_top_y(self, rect: pygame.Rect) -> int:
        """Return a stable top position for the score summary copy."""
        return rect.bottom - 76

    def _side_rows_bottom_y(self, rect: pygame.Rect) -> int:
        """Return the bottom y position after the side-panel metric rows."""
        row_count = 7
        rows_top_y = rect.y + 70 + len(PRESETS) * 38 + 16
        return rows_top_y + row_count * SIDE_ROW_STEP

    def _side_note_top_y(self, rect: pygame.Rect) -> int:
        """Return a stable top position for the side-panel explanatory note."""
        return rect.bottom - 70

    def _wrapped_text_bottom_y(
        self,
        text: str,
        width: int,
        font: pygame.font.Font,
        *,
        top_y: int,
        line_height: int,
    ) -> int:
        """Estimate wrapped text bottom for layout tests."""
        lines = 1
        current = ""
        for word in text.split():
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
                continue

            if current:
                lines += 1
            current = word

        return top_y + lines * line_height

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


def create_calibration_lab_scene(context: AppContext) -> CalibrationLabScene:
    """Create the unified shell Calibration Lab scene."""
    return CalibrationLabScene(context)
