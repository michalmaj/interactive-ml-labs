"""Native Class Imbalance Lab scene for the unified shell."""

from __future__ import annotations

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

THRESHOLDS: Final[tuple[float, ...]] = (0.30, 0.50, 0.70)
DEFAULT_THRESHOLD_INDEX: Final[int] = 1


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Static confusion counts for one threshold."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int


@dataclass(frozen=True, slots=True)
class ImbalancePreset:
    """Static class imbalance scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    positive_label_en: str
    positive_label_pl: str
    negative_count: int
    positive_count: int
    thresholds: tuple[ThresholdMetrics, ThresholdMetrics, ThresholdMetrics]

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

    def positive_label_for_language(self, language: str) -> str:
        """Return localized positive class label."""
        if language == "pl":
            return self.positive_label_pl
        return self.positive_label_en


PRESETS: Final[tuple[ImbalancePreset, ...]] = (
    ImbalancePreset(
        name_en="Fraud screening",
        name_pl="Fraud screening",
        summary_en="Fraud is rare, so accuracy can look strong while recall misses cases.",
        summary_pl="Fraud jest rzadki, więc accuracy może wyglądać dobrze mimo słabego recall.",
        positive_label_en="fraud",
        positive_label_pl="fraud",
        negative_count=940,
        positive_count=60,
        thresholds=(
            ThresholdMetrics(48, 70, 12, 870),
            ThresholdMetrics(33, 22, 27, 918),
            ThresholdMetrics(16, 5, 44, 935),
        ),
    ),
    ImbalancePreset(
        name_en="Medical triage",
        name_pl="Medical triage",
        summary_en="The risky class is uncommon, but missing it is expensive.",
        summary_pl="Ryzykowna klasa jest rzadka, ale jej przeoczenie dużo kosztuje.",
        positive_label_en="needs review",
        positive_label_pl="do sprawdzenia",
        negative_count=900,
        positive_count=100,
        thresholds=(
            ThresholdMetrics(86, 95, 14, 805),
            ThresholdMetrics(68, 38, 32, 862),
            ThresholdMetrics(42, 12, 58, 888),
        ),
    ),
    ImbalancePreset(
        name_en="Churn watchlist",
        name_pl="Churn watchlist",
        summary_en="The model must balance a useful watchlist with enough coverage.",
        summary_pl="Model musi zbalansować użyteczną watchlist z wystarczającym pokryciem.",
        positive_label_en="will churn",
        positive_label_pl="will churn",
        negative_count=820,
        positive_count=180,
        thresholds=(
            ThresholdMetrics(150, 150, 30, 670),
            ThresholdMetrics(112, 62, 68, 758),
            ThresholdMetrics(72, 20, 108, 800),
        ),
    ),
)


class ClassImbalanceLabScene:
    """Interactive slice for accuracy, precision, recall, and threshold trade-offs."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic imbalance scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.threshold_index = DEFAULT_THRESHOLD_INDEX

    @property
    def preset(self) -> ImbalancePreset:
        """Return the active imbalance preset."""
        return PRESETS[self.preset_index]

    @property
    def threshold(self) -> float:
        """Return the selected decision threshold."""
        return THRESHOLDS[self.threshold_index]

    @property
    def metrics(self) -> ThresholdMetrics:
        """Return the active confusion counts."""
        return self.preset.thresholds[self.threshold_index]

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
        """Draw the class imbalance lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_metrics_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.threshold_index = max(0, self.threshold_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.threshold_index = min(len(THRESHOLDS) - 1, self.threshold_index + 1)
        elif key in {pygame.K_0, pygame.K_KP0}:
            self.threshold_index = DEFAULT_THRESHOLD_INDEX
        elif key == pygame.K_r:
            self.preset_index = 0
            self.threshold_index = DEFAULT_THRESHOLD_INDEX

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Class Imbalance Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "See why high accuracy can hide poor minority-class recall.",
                "Zobacz, jak wysokie accuracy może ukrywać słaby recall klasy mniejszościowej.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_metrics_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Metrics at threshold", "Metryki przy threshold"),
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
        self._draw_metric_bars(surface, pygame.Rect(rect.x + 48, rect.y + 104, 280, 230))
        self._draw_confusion_matrix(surface, pygame.Rect(rect.x + 392, rect.y + 112, 246, 194))
        self._draw_class_mix(surface, pygame.Rect(rect.x + 58, rect.y + 372, rect.width - 116, 42))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 46),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        rows = (
            (self._label("preset", "preset"), self.preset.name_for_language(self._language)),
            (self._label("positive class", "klasa pozytywna"), self._positive_label()),
            (self._label("threshold", "threshold"), self._threshold_label()),
            (self._label("accuracy", "accuracy"), self._accuracy_label()),
            (self._label("precision", "precision"), self._precision_label()),
            (self._label("recall", "recall"), self._recall_label()),
            (self._label("false negatives", "false negatives"), str(self.metrics.false_negative)),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Read the score", "Czytaj wynik"),
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
                "Goal: choose a threshold by the mistakes that matter, not accuracy alone.",
                "Cel: wybieraj threshold przez ważne błędy, nie przez samo accuracy.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _total(self) -> int:
        return (
            self.metrics.true_positive
            + self.metrics.false_positive
            + self.metrics.false_negative
            + self.metrics.true_negative
        )

    def _accuracy(self) -> float:
        return (self.metrics.true_positive + self.metrics.true_negative) / self._total()

    def _precision(self) -> float:
        predicted_positive = self.metrics.true_positive + self.metrics.false_positive
        if predicted_positive == 0:
            return 0.0
        return self.metrics.true_positive / predicted_positive

    def _recall(self) -> float:
        actual_positive = self.metrics.true_positive + self.metrics.false_negative
        if actual_positive == 0:
            return 0.0
        return self.metrics.true_positive / actual_positive

    def _accuracy_label(self) -> str:
        return f"{self._accuracy():.0%}"

    def _precision_label(self) -> str:
        return f"{self._precision():.0%}"

    def _recall_label(self) -> str:
        return f"{self._recall():.0%}"

    def _threshold_label(self) -> str:
        return f"{self.threshold:.0%} ({self.threshold_index + 1}/{len(THRESHOLDS)})"

    def _positive_share_label(self) -> str:
        total = self.preset.positive_count + self.preset.negative_count
        return f"{self.preset.positive_count / total:.0%}"

    def _positive_label(self) -> str:
        return (
            f"{self.preset.positive_label_for_language(self._language)} "
            f"({self._positive_share_label()})"
        )

    def _diagnosis_key(self) -> str:
        if self._accuracy() >= 0.90 and self._recall() < 0.60:
            return "accuracy_trap"
        if self._recall() >= 0.80 and self._precision() < 0.50:
            return "wide_net"
        return "balanced"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "accuracy_trap":
            return self._label("accuracy trap", "pułapka accuracy")
        if diagnosis == "wide_net":
            return self._label("wide net", "szeroka sieć")
        return self._label("balanced trade-off", "rozsądny kompromis")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "accuracy_trap":
            return self._label(
                "Accuracy looks high, but recall is low. "
                "Inspect false negatives before trusting it.",
                "Accuracy wygląda wysoko, ale recall jest niski. Sprawdź false negatives.",
            )
        if diagnosis == "wide_net":
            return self._label(
                "Recall is strong, but many false positives may overload the review workflow.",
                "Recall jest mocny, ale wiele false positives może przeciążyć workflow.",
            )
        return self._label(
            "This threshold balances recall and precision better than accuracy alone suggests.",
            "Ten threshold lepiej balansuje recall i precision niż samo accuracy.",
        )

    def _draw_metric_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        metrics = (
            ("accuracy", self._accuracy(), ACCENT),
            ("precision", self._precision(), GOOD),
            ("recall", self._recall(), WARNING),
        )
        bar_width = 54
        gap = 28
        for index, (label, value, color) in enumerate(metrics):
            x = rect.x + 28 + index * (bar_width + gap)
            height = round((rect.height - 54) * value)
            bar = pygame.Rect(x, rect.bottom - 36 - height, bar_width, height)
            pygame.draw.rect(surface, color, bar, border_radius=6)
            self._draw_text(surface, f"{value:.0%}", (x - 2, bar.y - 24), self._font_small, TEXT)
            self._draw_text(
                surface,
                label,
                (x - 10, rect.bottom - 24),
                self._font_small,
                MUTED_TEXT,
            )

    def _draw_confusion_matrix(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_text(
            surface,
            self._label("Confusion matrix", "Confusion matrix"),
            (rect.x, rect.y - 30),
            self._font_small,
            TEXT,
        )
        cells = (
            ("TP", self.metrics.true_positive, GOOD),
            ("FP", self.metrics.false_positive, SECONDARY),
            ("FN", self.metrics.false_negative, WARNING),
            ("TN", self.metrics.true_negative, ACCENT),
        )
        cell_w = (rect.width - 12) // 2
        cell_h = (rect.height - 12) // 2
        for index, (label, value, color) in enumerate(cells):
            x = rect.x + (index % 2) * (cell_w + 12)
            y = rect.y + (index // 2) * (cell_h + 12)
            cell = pygame.Rect(x, y, cell_w, cell_h)
            pygame.draw.rect(surface, PLOT_BG, cell, border_radius=6)
            pygame.draw.rect(surface, color, cell, width=2, border_radius=6)
            self._draw_text(surface, label, (x + 12, y + 10), self._font_small, color)
            self._draw_text(surface, str(value), (x + 12, y + 38), self._font_body, TEXT)

    def _draw_class_mix(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        total = self.preset.positive_count + self.preset.negative_count
        positive_width = round(rect.width * self.preset.positive_count / total)
        negative_width = rect.width - positive_width
        pygame.draw.rect(
            surface,
            ACCENT,
            pygame.Rect(rect.x, rect.y, negative_width, 18),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            WARNING,
            pygame.Rect(rect.x + negative_width, rect.y, positive_width, 18),
            border_radius=5,
        )
        self._draw_text(
            surface,
            self._label(
                f"minority class: {self.preset.positive_count}/{total}",
                f"klasa mniejszościowa: {self.preset.positive_count}/{total}",
            ),
            (rect.x, rect.y + 24),
            self._font_small,
            MUTED_TEXT,
        )

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


def create_class_imbalance_lab_scene(context: AppContext) -> ClassImbalanceLabScene:
    """Create the unified shell Class Imbalance Lab scene."""
    return ClassImbalanceLabScene(context)
