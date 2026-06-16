"""Native Feature Scaling Lab scene for the unified shell."""

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

MODELS: Final[tuple[str, ...]] = ("k-NN", "Logistic Regression", "Gradient Descent")


@dataclass(frozen=True, slots=True)
class ScalingScores:
    """Scores before and after scaling for one model."""

    raw_accuracy: float
    scaled_accuracy: float
    raw_iterations: int
    scaled_iterations: int


@dataclass(frozen=True, slots=True)
class ScalingPreset:
    """Static feature scaling scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    feature_a_en: str
    feature_a_pl: str
    feature_b_en: str
    feature_b_pl: str
    raw_range_a: int
    raw_range_b: int
    scaled_range_a: int
    scaled_range_b: int
    scores: tuple[ScalingScores, ScalingScores, ScalingScores]

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

    def feature_a_for_language(self, language: str) -> str:
        """Return localized first feature name."""
        if language == "pl":
            return self.feature_a_pl
        return self.feature_a_en

    def feature_b_for_language(self, language: str) -> str:
        """Return localized second feature name."""
        if language == "pl":
            return self.feature_b_pl
        return self.feature_b_en


PRESETS: Final[tuple[ScalingPreset, ...]] = (
    ScalingPreset(
        name_en="Income vs age",
        name_pl="Income vs age",
        summary_en="Income has a huge numeric range, so distance-based models overuse it.",
        summary_pl="Income ma ogromny zakres liczbowy, więc modele dystansowe używają go za mocno.",
        feature_a_en="income",
        feature_a_pl="income",
        feature_b_en="age",
        feature_b_pl="age",
        raw_range_a=120_000,
        raw_range_b=60,
        scaled_range_a=1,
        scaled_range_b=1,
        scores=(
            ScalingScores(0.63, 0.82, 42, 18),
            ScalingScores(0.70, 0.81, 64, 26),
            ScalingScores(0.58, 0.79, 220, 72),
        ),
    ),
    ScalingPreset(
        name_en="Pixel intensity vs count",
        name_pl="Pixel intensity vs count",
        summary_en="One feature is already bounded, the other grows with object size.",
        summary_pl="Jedna cecha jest ograniczona, druga rośnie razem z rozmiarem obiektu.",
        feature_a_en="object count",
        feature_a_pl="object count",
        feature_b_en="mean intensity",
        feature_b_pl="mean intensity",
        raw_range_a=900,
        raw_range_b=1,
        scaled_range_a=1,
        scaled_range_b=1,
        scores=(
            ScalingScores(0.66, 0.79, 38, 20),
            ScalingScores(0.72, 0.78, 56, 31),
            ScalingScores(0.61, 0.77, 180, 82),
        ),
    ),
    ScalingPreset(
        name_en="Sensor units",
        name_pl="Jednostki sensorów",
        summary_en="Different physical units make raw coefficients and distances misleading.",
        summary_pl="Różne jednostki fizyczne psują interpretację dystansów i coefficients.",
        feature_a_en="pressure",
        feature_a_pl="pressure",
        feature_b_en="temperature",
        feature_b_pl="temperature",
        raw_range_a=5_000,
        raw_range_b=45,
        scaled_range_a=1,
        scaled_range_b=1,
        scores=(
            ScalingScores(0.65, 0.80, 46, 19),
            ScalingScores(0.69, 0.82, 70, 30),
            ScalingScores(0.55, 0.78, 260, 86),
        ),
    ),
)


class FeatureScalingLabScene:
    """Interactive slice for showing how scaling changes model behavior."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic feature scaling scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.model_index = 0
        self.scaling_enabled = False

    @property
    def preset(self) -> ScalingPreset:
        """Return the active scaling preset."""
        return PRESETS[self.preset_index]

    @property
    def model_name(self) -> str:
        """Return the active model name."""
        return MODELS[self.model_index]

    @property
    def scores(self) -> ScalingScores:
        """Return scores for the active model."""
        return self.preset.scores[self.model_index]

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
        """Draw the feature scaling lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_scale_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key == pygame.K_s:
            self.scaling_enabled = not self.scaling_enabled
        elif key == pygame.K_m:
            self.model_index = (self.model_index + 1) % len(MODELS)
        elif key == pygame.K_r:
            self.preset_index = 0
            self.model_index = 0
            self.scaling_enabled = False

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Feature Scaling Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Watch one feature dominate until scaling makes ranges comparable.",
                "Zobacz, jak jedna cecha dominuje, dopóki scaling nie wyrówna zakresów.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_scale_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Feature ranges", "Zakresy cech"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._scaling_state_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            GOOD if self.scaling_enabled else WARNING,
        )
        self._draw_range_bars(surface, pygame.Rect(rect.x + 54, rect.y + 118, 588, 142))
        self._draw_metric_bars(surface, pygame.Rect(rect.x + 96, rect.y + 314, 500, 86))
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
            (self._label("model", "model"), self.model_name),
            (self._label("scaling", "scaling"), self._scaling_state_label()),
            (self._label("accuracy", "accuracy"), self._accuracy_label()),
            (self._label("iterations", "iterations"), self._iterations_label()),
            (self._label("range ratio", "range ratio"), self._range_ratio_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Scaling check", "Kontrola scaling"),
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
                "Goal: compare models after making feature ranges comparable.",
                "Cel: porównuj modele dopiero po wyrównaniu zakresów cech.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _accuracy(self) -> float:
        return self.scores.scaled_accuracy if self.scaling_enabled else self.scores.raw_accuracy

    def _iterations(self) -> int:
        return self.scores.scaled_iterations if self.scaling_enabled else self.scores.raw_iterations

    def _accuracy_label(self) -> str:
        return f"{self._accuracy():.0%}"

    def _iterations_label(self) -> str:
        return str(self._iterations())

    def _range_values(self) -> tuple[int, int]:
        if self.scaling_enabled:
            return (self.preset.scaled_range_a, self.preset.scaled_range_b)
        return (self.preset.raw_range_a, self.preset.raw_range_b)

    def _range_ratio(self) -> int:
        first, second = self._range_values()
        return max(first, second) // max(1, min(first, second))

    def _range_ratio_label(self) -> str:
        return f"{self._range_ratio()}:1"

    def _scaling_state_label(self) -> str:
        if self.scaling_enabled:
            return self._label("scaled", "scaled")
        return self._label("raw ranges", "raw ranges")

    def _diagnosis_key(self) -> str:
        if self.scaling_enabled:
            return "comparable"
        if self._range_ratio() >= 50:
            return "dominance"
        return "review"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "comparable":
            return self._label("comparable ranges", "porównywalne zakresy")
        if diagnosis == "dominance":
            return self._label("feature dominance", "dominacja cechy")
        return self._label("review scale", "sprawdź skalę")

    def _active_takeaway(self) -> str:
        if self.scaling_enabled:
            return self._label(
                "Ranges are comparable now, so distance and gradients are less distorted.",
                "Zakresy są porównywalne, więc dystans i gradient są mniej zniekształcone.",
            )
        return self._label(
            "One numeric range dominates. Press S before comparing model behavior.",
            "Jeden zakres liczbowy dominuje. Naciśnij S przed porównaniem modeli.",
        )

    def _draw_range_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        labels = (
            self.preset.feature_a_for_language(self._language),
            self.preset.feature_b_for_language(self._language),
        )
        values = self._range_values()
        max_value = max(values)
        for index, (label, value, color) in enumerate(
            zip(labels, values, (ACCENT, SECONDARY), strict=True)
        ):
            y = rect.y + index * 66
            self._draw_text(surface, label, (rect.x, y), self._font_small, TEXT)
            pygame.draw.rect(
                surface,
                GRID,
                pygame.Rect(rect.x, y + 26, rect.width, 16),
                border_radius=5,
            )
            fill_width = max(8, round(rect.width * value / max_value))
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(rect.x, y + 26, fill_width, 16),
                border_radius=5,
            )
            self._draw_text(surface, str(value), (rect.right - 80, y), self._font_small, MUTED_TEXT)

    def _draw_metric_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        raw_width = round(rect.width * self.scores.raw_accuracy)
        scaled_width = round(rect.width * self.scores.scaled_accuracy)
        pygame.draw.rect(
            surface,
            GRID,
            pygame.Rect(rect.x, rect.y, rect.width, 16),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            WARNING,
            pygame.Rect(rect.x, rect.y, raw_width, 16),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            GRID,
            pygame.Rect(rect.x, rect.y + 44, rect.width, 16),
            border_radius=5,
        )
        pygame.draw.rect(
            surface,
            GOOD,
            pygame.Rect(rect.x, rect.y + 44, scaled_width, 16),
            border_radius=5,
        )
        self._draw_text(surface, "raw", (rect.x, rect.y - 22), self._font_small, WARNING)
        self._draw_text(
            surface,
            f"{self.scores.raw_accuracy:.0%}",
            (rect.right - 54, rect.y - 22),
            self._font_small,
            MUTED_TEXT,
        )
        self._draw_text(surface, "scaled", (rect.x, rect.y + 22), self._font_small, GOOD)
        self._draw_text(
            surface,
            f"{self.scores.scaled_accuracy:.0%}",
            (rect.right - 54, rect.y + 22),
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


def create_feature_scaling_lab_scene(context: AppContext) -> FeatureScalingLabScene:
    """Create the unified shell Feature Scaling Lab scene."""
    return FeatureScalingLabScene(context)
