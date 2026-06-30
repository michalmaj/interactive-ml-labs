"""Native Train / Validation / Test Split Lab scene for the unified shell."""

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

DEFAULT_COMPLEXITY_INDEX: Final[int] = 1
COMPLEXITY_LABELS: Final[tuple[str, str, str]] = ("simple", "balanced", "too flexible")
SPLIT_LESSON_ID: Final[str] = "trustworthy_split"
COMPARE_COMPLEXITY_TASK_ID: Final[str] = "compare_complexity_settings"
CHOOSE_VALIDATION_TASK_ID: Final[str] = "choose_validation_candidate"


@dataclass(frozen=True, slots=True)
class SplitMetrics:
    """Static train/validation/test scores for one model complexity."""

    train: float
    validation: float
    test: float


@dataclass(frozen=True, slots=True)
class SplitPreset:
    """Static split scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    split_counts: tuple[int, int, int]
    metrics: tuple[SplitMetrics, SplitMetrics, SplitMetrics]

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


PRESETS: Final[tuple[SplitPreset, ...]] = (
    SplitPreset(
        name_en="Clean signal",
        name_pl="Czysty sygnał",
        summary_en="Validation and test agree when the split is representative.",
        summary_pl="Validation i test są podobne, gdy split jest reprezentatywny.",
        split_counts=(600, 200, 200),
        metrics=(
            SplitMetrics(0.74, 0.72, 0.71),
            SplitMetrics(0.88, 0.84, 0.83),
            SplitMetrics(0.97, 0.78, 0.76),
        ),
    ),
    SplitPreset(
        name_en="Noisy labels",
        name_pl="Szum w etykietach",
        summary_en="A flexible model memorizes training noise and loses validation quality.",
        summary_pl="Elastyczny model zapamiętuje noise z train i traci jakość na validation.",
        split_counts=(600, 200, 200),
        metrics=(
            SplitMetrics(0.66, 0.64, 0.63),
            SplitMetrics(0.82, 0.75, 0.74),
            SplitMetrics(0.99, 0.67, 0.65),
        ),
    ),
    SplitPreset(
        name_en="Small dataset",
        name_pl="Mały dataset",
        summary_en="With little data, validation is noisier but still protects the test set.",
        summary_pl=(
            "Przy małej liczbie danych validation jest bardziej szumne, ale chroni test set."
        ),
        split_counts=(120, 40, 40),
        metrics=(
            SplitMetrics(0.70, 0.65, 0.64),
            SplitMetrics(0.86, 0.76, 0.73),
            SplitMetrics(1.00, 0.61, 0.58),
        ),
    ),
)


class TrainValidationTestLabScene:
    """Interactive slice for selecting complexity with validation, not test."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic split lab scene."""
        self._context = context
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.complexity_index = DEFAULT_COMPLEXITY_INDEX
        self._seen_complexity_indices = {self.complexity_index}

    @property
    def preset(self) -> SplitPreset:
        """Return the active split preset."""
        return PRESETS[self.preset_index]

    @property
    def metrics(self) -> SplitMetrics:
        """Return scores for the selected complexity."""
        return self.preset.metrics[self.complexity_index]

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
        """Draw the split lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_score_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.complexity_index = max(0, self.complexity_index - 1)
            self._record_complexity_progress()
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.complexity_index = min(len(COMPLEXITY_LABELS) - 1, self.complexity_index + 1)
            self._record_complexity_progress()
        elif key in {pygame.K_0, pygame.K_KP0}:
            self.complexity_index = DEFAULT_COMPLEXITY_INDEX
            self._record_complexity_progress()
        elif key == pygame.K_r:
            self.preset_index = 0
            self.complexity_index = DEFAULT_COMPLEXITY_INDEX
            self._seen_complexity_indices = {self.complexity_index}

    def _record_complexity_progress(self) -> None:
        """Record meaningful complexity comparison for the guided lesson."""
        if self._context.selected_lesson_id != SPLIT_LESSON_ID:
            return

        self._seen_complexity_indices.add(self.complexity_index)
        if len(self._seen_complexity_indices) >= 2:
            self._context.progress.complete_task(
                SPLIT_LESSON_ID,
                COMPARE_COMPLEXITY_TASK_ID,
            )

            if self.complexity_index == self._best_validation_complexity_index():
                self._context.progress.complete_task(
                    SPLIT_LESSON_ID,
                    CHOOSE_VALIDATION_TASK_ID,
                )

        self._mark_lesson_completed_if_ready()

    def _best_validation_complexity_index(self) -> int:
        """Return the complexity with the strongest validation score."""
        return max(
            range(len(self.preset.metrics)),
            key=lambda index: self.preset.metrics[index].validation,
        )

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both guided tasks are done."""
        progress = self._context.progress.lessons.get(SPLIT_LESSON_ID)
        if progress is None:
            return

        required_tasks = {COMPARE_COMPLEXITY_TASK_ID, CHOOSE_VALIDATION_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(SPLIT_LESSON_ID)

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            "Train / Validation / Test Split Lab",
            (58, 40),
            self._font_title,
            TEXT,
        )
        self._draw_text(
            surface,
            self._label(
                "Tune on validation, keep test as the final honest check.",
                "Dobieraj model na validation, a test zostaw na uczciwy finał.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_score_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Scores by split", "Wyniki według splitu"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._complexity_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        self._draw_score_bars(surface, pygame.Rect(rect.x + 58, rect.y + 116, 360, 220))
        self._draw_split_mix(surface, pygame.Rect(rect.x + 470, rect.y + 132, 150, 180))
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 52),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        rows = (
            (self._label("preset", "preset"), self.preset.name_for_language(self._language)),
            (self._label("complexity", "complexity"), self._complexity_label()),
            (self._label("train score", "train score"), self._score_label(self.metrics.train)),
            (
                self._label("validation score", "validation score"),
                self._score_label(self.metrics.validation),
            ),
            (self._label("test score", "test score"), self._score_label(self.metrics.test)),
            (self._label("train-val gap", "train-val gap"), self._gap_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Model selection", "Wybór modelu"),
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
                "Goal: use validation for choices, test for the last unbiased estimate.",
                "Cel: validation służy do wyboru, test do ostatniej niezależnej oceny.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _complexity_label(self) -> str:
        label = COMPLEXITY_LABELS[self.complexity_index]
        if self._language == "pl":
            labels = ("prosty", "zbalansowany", "zbyt elastyczny")
            label = labels[self.complexity_index]
        return f"{label} ({self.complexity_index + 1}/{len(COMPLEXITY_LABELS)})"

    def _score_label(self, value: float) -> str:
        return f"{value:.0%}"

    def _gap(self) -> float:
        return self.metrics.train - self.metrics.validation

    def _gap_label(self) -> str:
        return f"+{self._gap():.0%}"

    def _diagnosis_key(self) -> str:
        if self.metrics.validation < 0.70:
            return "underfit"
        if self._gap() >= 0.18:
            return "overfit"
        return "candidate"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "underfit":
            return self._label("underfit", "underfit")
        if diagnosis == "overfit":
            return self._label("overfit", "overfit")
        return self._label("validation candidate", "kandydat z validation")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "underfit":
            return self._label(
                "Validation is weak too, so the model is too simple for the signal.",
                "Validation też jest słabe, więc model jest zbyt prosty dla sygnału.",
            )
        if diagnosis == "overfit":
            return self._label(
                "Train score is high, but validation drops. Do not choose by train score.",
                "Train score jest wysoki, ale validation spada. Nie wybieraj po train score.",
            )
        return self._label(
            "Validation is strongest here. Touch the test set only after choosing the model.",
            "Validation jest tu najmocniejsze. Test set sprawdź dopiero po wyborze modelu.",
        )

    def _draw_score_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        scores = (
            ("train", self.metrics.train, ACCENT),
            ("validation", self.metrics.validation, GOOD),
            ("test", self.metrics.test, SECONDARY),
        )
        bar_width = 66
        gap = 38
        for index, (label, value, color) in enumerate(scores):
            x = rect.x + 42 + index * (bar_width + gap)
            height = round((rect.height - 54) * value)
            bar = pygame.Rect(x, rect.bottom - 36 - height, bar_width, height)
            pygame.draw.rect(surface, color, bar, border_radius=6)
            self._draw_text(surface, f"{value:.0%}", (x + 6, bar.y - 24), self._font_small, TEXT)
            self._draw_text(surface, label, (x - 8, rect.bottom - 24), self._font_small, MUTED_TEXT)

    def _draw_split_mix(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_text(
            surface,
            self._label("split sizes", "rozmiary splitów"),
            (rect.x, rect.y - 30),
            self._font_small,
            TEXT,
        )
        labels = ("train", "val", "test")
        colors = (ACCENT, GOOD, SECONDARY)
        total = sum(self.preset.split_counts)
        y = rect.y
        for label, count, color in zip(labels, self.preset.split_counts, colors, strict=True):
            width = round(rect.width * count / total)
            pygame.draw.rect(surface, GRID, pygame.Rect(rect.x, y, rect.width, 18), border_radius=5)
            pygame.draw.rect(surface, color, pygame.Rect(rect.x, y, width, 18), border_radius=5)
            self._draw_text(
                surface,
                f"{label}: {count}",
                (rect.x, y + 24),
                self._font_small,
                MUTED_TEXT,
            )
            y += 56

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


def create_train_validation_test_lab_scene(context: AppContext) -> TrainValidationTestLabScene:
    """Create the unified shell Train / Validation / Test Split Lab scene."""
    return TrainValidationTestLabScene(context)
