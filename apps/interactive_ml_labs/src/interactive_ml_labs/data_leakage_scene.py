"""Native Data Leakage Lab scene for the unified shell."""

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
LEAKAGE_LESSON_ID: Final[str] = "trustworthy_leakage"
REMOVE_LEAKAGE_TASK_ID: Final[str] = "remove_leakage_feature"
COMPARE_SCENARIOS_TASK_ID: Final[str] = "compare_leakage_scenarios"


@dataclass(frozen=True, slots=True)
class LeakagePreset:
    """Static leakage scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    leak_feature_en: str
    leak_feature_pl: str
    clean_train: float
    clean_test: float
    leaky_train: float
    leaky_test: float
    feature_scores: tuple[float, float, float]

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

    def leak_feature_for_language(self, language: str) -> str:
        """Return localized suspicious feature name."""
        if language == "pl":
            return self.leak_feature_pl
        return self.leak_feature_en


PRESETS: Final[tuple[LeakagePreset, ...]] = (
    LeakagePreset(
        name_en="Post-outcome column",
        name_pl="Kolumna po wyniku",
        summary_en="A feature is filled only after the outcome is already known.",
        summary_pl="Cecha powstaje dopiero po tym, gdy wynik jest juz znany.",
        leak_feature_en="approved_at timestamp",
        leak_feature_pl="timestamp approved_at",
        clean_train=0.74,
        clean_test=0.71,
        leaky_train=0.99,
        leaky_test=0.98,
        feature_scores=(0.22, 0.18, 0.96),
    ),
    LeakagePreset(
        name_en="Target proxy",
        name_pl="Proxy targetu",
        summary_en="One feature is almost the target label written under another name.",
        summary_pl="Jedna cecha jest prawie etykietą target, tylko pod inną nazwą.",
        leak_feature_en="refund_status",
        leak_feature_pl="refund_status",
        clean_train=0.69,
        clean_test=0.66,
        leaky_train=0.96,
        leaky_test=0.94,
        feature_scores=(0.26, 0.21, 0.91),
    ),
    LeakagePreset(
        name_en="Future aggregate",
        name_pl="Agregat z przyszłości",
        summary_en=(
            "A rolling feature accidentally includes future events from after prediction time."
        ),
        summary_pl="Rolling feature przypadkiem zawiera przyszłe zdarzenia po czasie predykcji.",
        leak_feature_en="next_7d_activity",
        leak_feature_pl="next_7d_activity",
        clean_train=0.76,
        clean_test=0.72,
        leaky_train=0.93,
        leaky_test=0.90,
        feature_scores=(0.30, 0.24, 0.84),
    ),
)


class DataLeakageLabScene:
    """Interactive slice for spotting suspicious leakage-driven scores."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic leakage scene."""
        self._context = context
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.leakage_enabled = True
        self._seen_preset_indices = {self.preset_index}

    @property
    def preset(self) -> LeakagePreset:
        """Return the active leakage preset."""
        return PRESETS[self.preset_index]

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
        """Draw the leakage lab."""
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
            self._record_scenario_progress()
        elif key == pygame.K_l:
            self.leakage_enabled = not self.leakage_enabled
            self._record_leakage_progress()
        elif key == pygame.K_r:
            self.preset_index = 0
            self.leakage_enabled = True
            self._seen_preset_indices = {self.preset_index}

    def _record_scenario_progress(self) -> None:
        """Complete scenario comparison after visiting a second preset."""
        if self._context.selected_lesson_id != LEAKAGE_LESSON_ID:
            return

        self._seen_preset_indices.add(self.preset_index)
        if len(self._seen_preset_indices) >= 2:
            self._context.progress.complete_task(
                LEAKAGE_LESSON_ID,
                COMPARE_SCENARIOS_TASK_ID,
            )
        self._mark_lesson_completed_if_ready()

    def _record_leakage_progress(self) -> None:
        """Complete leakage removal after excluding the suspicious feature."""
        if self._context.selected_lesson_id != LEAKAGE_LESSON_ID:
            return

        if not self.leakage_enabled:
            self._context.progress.complete_task(
                LEAKAGE_LESSON_ID,
                REMOVE_LEAKAGE_TASK_ID,
            )
        self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both guided tasks are done."""
        progress = self._context.progress.lessons.get(LEAKAGE_LESSON_ID)
        if progress is None:
            return

        required_tasks = {REMOVE_LEAKAGE_TASK_ID, COMPARE_SCENARIOS_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(LEAKAGE_LESSON_ID)

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Data Leakage Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Find the feature that makes validation look too good to be true.",
                "Znajdź cechę, przez którą walidacja wygląda podejrzanie dobrze.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_score_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Score check", "Kontrola wyniku"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._model_state_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            WARNING if self.leakage_enabled else GOOD,
        )
        bar_rect = pygame.Rect(rect.x + 80, rect.y + 116, rect.width - 160, 194)
        self._draw_accuracy_bars(surface, bar_rect)
        feature_rect = pygame.Rect(rect.x + 58, rect.y + 350, rect.width - 116, 78)
        self._draw_feature_scores(surface, feature_rect)
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 24, rect.bottom - 50),
            rect.width - 48,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        rows = (
            (self._label("preset", "preset"), self.preset.name_for_language(self._language)),
            (self._label("leakage feature", "cecha leakage"), self._leakage_feature_label()),
            (self._label("train accuracy", "train accuracy"), self._train_accuracy_label()),
            (self._label("test accuracy", "test accuracy"), self._test_accuracy_label()),
            (self._label("gap", "gap"), self._gap_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Leakage checklist", "Checklist leakage"),
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
                "Goal: learn to distrust perfect metrics before trusting the model.",
                "Cel: naucz się nie ufać idealnym metrykom bez sprawdzenia danych.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _train_accuracy(self) -> float:
        return self.preset.leaky_train if self.leakage_enabled else self.preset.clean_train

    def _test_accuracy(self) -> float:
        return self.preset.leaky_test if self.leakage_enabled else self.preset.clean_test

    def _generalization_gap(self) -> float:
        return self._train_accuracy() - self._test_accuracy()

    def _train_accuracy_label(self) -> str:
        return f"{self._train_accuracy():.0%}"

    def _test_accuracy_label(self) -> str:
        return f"{self._test_accuracy():.0%}"

    def _gap_label(self) -> str:
        return f"{self._generalization_gap():.0%}"

    def _leakage_feature_label(self) -> str:
        if self.leakage_enabled:
            return self.preset.leak_feature_for_language(self._language)
        return self._label("removed", "usunięta")

    def _model_state_label(self) -> str:
        if self.leakage_enabled:
            return self._label("leakage feature included", "cecha leakage włączona")
        return self._label("leakage feature removed", "cecha leakage usunięta")

    def _diagnosis_key(self) -> str:
        if (
            self.leakage_enabled
            and self._test_accuracy() >= 0.90
            and self._generalization_gap() < 0.04
        ):
            return "too_good"
        if self._test_accuracy() < 0.78:
            return "realistic"
        return "review"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "too_good":
            return self._label("too good to trust", "zbyt dobre, by ufać")
        if diagnosis == "realistic":
            return self._label("more realistic", "bardziej realistyczne")
        return self._label("review features", "sprawdź cechy")

    def _active_takeaway(self) -> str:
        if self.leakage_enabled:
            return self._label(
                "The score is almost perfect. Ask whether the feature exists at prediction time.",
                "Wynik jest prawie idealny. Sprawdź, czy ta cecha istnieje w czasie predykcji.",
            )
        return self._label(
            "After removing leakage, the score drops but becomes useful for honest validation.",
            "Po usunięciu leakage wynik spada, ale lepiej nadaje się do uczciwej walidacji.",
        )

    def _draw_accuracy_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            y = rect.bottom - round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)
        self._draw_metric_bar(surface, rect, 0, self._train_accuracy(), "train", ACCENT)
        self._draw_metric_bar(surface, rect, 1, self._test_accuracy(), "test", SECONDARY)

    def _draw_metric_bar(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        index: int,
        value: float,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        width = 104
        gap = 70
        x = rect.centerx - width - gap // 2 + index * (width + gap)
        height = round(rect.height * value)
        bar = pygame.Rect(x, rect.bottom - height, width, height)
        pygame.draw.rect(surface, color, bar, border_radius=6)
        self._draw_text(surface, f"{value:.0%}", (x + 28, bar.y - 28), self._font_small, TEXT)
        self._draw_text(surface, label, (x + 30, rect.bottom + 10), self._font_small, MUTED_TEXT)

    def _draw_feature_scores(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        labels = (
            self._label("real signal", "real signal"),
            self._label("noise", "noise"),
            self._label("leakage", "leakage"),
        )
        scores = self.preset.feature_scores
        bar_width = (rect.width - 52) // 3
        for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
            x = rect.x + index * (bar_width + 26)
            color = WARNING if index == 2 and self.leakage_enabled else MUTED_TEXT
            self._draw_text(surface, label, (x, rect.y), self._font_small, color)
            pygame.draw.rect(
                surface,
                GRID,
                pygame.Rect(x, rect.y + 28, bar_width, 12),
                border_radius=4,
            )
            if index < 2 or self.leakage_enabled:
                fill_width = round(bar_width * score)
                pygame.draw.rect(
                    surface,
                    color,
                    pygame.Rect(x, rect.y + 28, fill_width, 12),
                    border_radius=4,
                )
            self._draw_text(surface, f"{score:.0%}", (x, rect.y + 48), self._font_small, MUTED_TEXT)

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


def create_data_leakage_lab_scene(context: AppContext) -> DataLeakageLabScene:
    """Create the unified shell Data Leakage Lab scene."""
    return DataLeakageLabScene(context)
