"""Native Feature Importance Lab scene for the unified shell."""

from __future__ import annotations

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


class ImportanceMethod(StrEnum):
    """Feature importance method shown in the lab."""

    PERMUTATION = "permutation"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class FeatureSignal:
    """Static feature importance row."""

    name: str
    permutation: float
    model: float
    stability: float
    correlation_group: str
    leakage_risk: bool = False


@dataclass(frozen=True, slots=True)
class ImportancePreset:
    """Feature importance scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    features: tuple[FeatureSignal, ...]

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


PRESETS: Final[tuple[ImportancePreset, ...]] = (
    ImportancePreset(
        name_en="Correlated features",
        name_pl="Skorelowane cechy",
        summary_en=(
            "Two related features share signal, so one importance bar can shrink "
            "even though the feature still matters."
        ),
        summary_pl=(
            "Dwie powiązane cechy dzielą sygnał, więc jeden importance bar może "
            "zmalać, mimo że cecha nadal ma znaczenie."
        ),
        features=(
            FeatureSignal("income", 0.31, 0.46, 0.76, "A"),
            FeatureSignal("credit_limit", 0.18, 0.38, 0.62, "A"),
            FeatureSignal("age", 0.14, 0.10, 0.88, "B"),
            FeatureSignal("region", 0.08, 0.06, 0.72, "C"),
            FeatureSignal("signup_day", 0.03, 0.02, 0.41, "D"),
        ),
    ),
    ImportancePreset(
        name_en="Leakage candidate",
        name_pl="Podejrzenie leakage",
        summary_en=(
            "A suspicious future-looking feature dominates the ranking and should be "
            "removed before trusting the model."
        ),
        summary_pl=(
            "Podejrzana cecha z przyszłości dominuje ranking i trzeba ją usunąć, "
            "zanim zaufasz modelowi."
        ),
        features=(
            FeatureSignal("future_status", 0.58, 0.64, 0.95, "leak", leakage_risk=True),
            FeatureSignal("recent_activity", 0.18, 0.16, 0.82, "A"),
            FeatureSignal("plan_type", 0.10, 0.09, 0.78, "B"),
            FeatureSignal("support_tickets", 0.07, 0.05, 0.68, "C"),
            FeatureSignal("country", 0.03, 0.04, 0.55, "D"),
        ),
    ),
    ImportancePreset(
        name_en="Unstable ranking",
        name_pl="Niestabilny ranking",
        summary_en=(
            "Several weak signals trade places across folds, so the exact order is less "
            "important than the group."
        ),
        summary_pl=(
            "Kilka słabszych sygnałów zamienia się miejscami między foldami, więc "
            "dokładna kolejność jest mniej ważna niż grupa."
        ),
        features=(
            FeatureSignal("session_count", 0.24, 0.20, 0.52, "A"),
            FeatureSignal("avg_duration", 0.21, 0.28, 0.48, "A"),
            FeatureSignal("last_seen_days", 0.19, 0.15, 0.44, "B"),
            FeatureSignal("device_count", 0.16, 0.18, 0.50, "C"),
            FeatureSignal("campaign", 0.05, 0.06, 0.37, "D"),
        ),
    ),
)


class FeatureImportanceLabScene:
    """Interactive feature importance interpretation lab."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic feature importance lab."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.method = ImportanceMethod.PERMUTATION
        self.show_correlation_groups = True
        self.show_leakage_warning = True

    @property
    def preset(self) -> ImportancePreset:
        """Return the active feature importance preset."""
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
        """Draw the feature importance lab."""
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
            self.method = (
                ImportanceMethod.MODEL
                if self.method == ImportanceMethod.PERMUTATION
                else ImportanceMethod.PERMUTATION
            )
        elif key == pygame.K_c:
            self.show_correlation_groups = not self.show_correlation_groups
        elif key == pygame.K_l:
            self.show_leakage_warning = not self.show_leakage_warning
        elif key == pygame.K_r:
            self.preset_index = 0
            self.method = ImportanceMethod.PERMUTATION
            self.show_correlation_groups = True
            self.show_leakage_warning = True

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Feature Importance Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Compare importance methods without confusing importance with causality.",
                "Porównuj metody importance bez mylenia importance z przyczynowością.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_chart_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Importance ranking", "Ranking importance"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._method_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        self._draw_importance_bars(surface, pygame.Rect(rect.x + 44, rect.y + 104, 612, 246))
        self._draw_stability_strip(surface, pygame.Rect(rect.x + 64, rect.y + 390, 570, 38))
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
        top_feature = self.top_feature()
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("method", "metoda"), self._method_label()),
            (self._label("top feature", "top feature"), top_feature.name),
            (self._label("top score", "top score"), f"{self._score_for(top_feature):.0%}"),
            (self._label("stability", "stability"), f"{top_feature.stability:.0%}"),
            (self._label("correlation", "korelacja"), self._correlation_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Interpretation", "Interpretacja"),
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
                "1-3: dataset | M: method | C: correlation groups | L: leakage warning | R: reset",
                "1-3: dataset | M: metoda | C: korelacje | L: leakage warning | R: reset",
            ),
            (58, 642),
            self._font_small,
            TEXT,
        )

    def _draw_importance_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        sorted_features = self.sorted_features()
        max_score = max(self._score_for(feature) for feature in sorted_features)
        row_height = 38
        for index, feature in enumerate(sorted_features):
            y = rect.y + (index * 45)
            score = self._score_for(feature)
            width = round((rect.width - 190) * score / max(max_score, 0.001))
            color = WARNING if feature.leakage_risk and self.show_leakage_warning else ACCENT
            label_color = SECONDARY if self._is_correlated(feature) else TEXT
            bar_back = pygame.Rect(rect.x + 170, y, rect.width - 190, row_height)
            bar_fill = pygame.Rect(rect.x + 170, y, width, row_height)
            self._draw_text(surface, feature.name, (rect.x, y + 9), self._font_small, label_color)
            pygame.draw.rect(surface, PLOT_BG, bar_back, border_radius=5)
            pygame.draw.rect(surface, color, bar_fill, border_radius=5)
            pygame.draw.rect(surface, GRID, bar_back, width=1, border_radius=5)
            self._draw_text(
                surface,
                f"{score:.0%}",
                (rect.right - 44, y + 9),
                self._font_small,
                TEXT,
            )
            if feature.leakage_risk and self.show_leakage_warning:
                self._draw_text(
                    surface,
                    "leakage?",
                    (rect.x + 88, y + 9),
                    self._font_small,
                    WARNING,
                )

    def _draw_stability_strip(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_text(surface, "stability", (rect.x, rect.y - 22), self._font_small, MUTED_TEXT)
        segment_width = rect.width // len(self.preset.features)
        for index, feature in enumerate(self.preset.features):
            x = rect.x + (index * segment_width)
            height = round(rect.height * feature.stability)
            y = rect.bottom - height
            if feature.stability >= 0.7:
                color = GOOD
            elif feature.stability >= 0.5:
                color = SECONDARY
            else:
                color = WARNING
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(x + 5, y, segment_width - 10, height),
                border_radius=4,
            )

    def sorted_features(self) -> tuple[FeatureSignal, ...]:
        """Return features sorted by active importance score."""
        return tuple(sorted(self.preset.features, key=self._score_for, reverse=True))

    def top_feature(self) -> FeatureSignal:
        """Return the top-ranked feature for the active method."""
        return self.sorted_features()[0]

    def _score_for(self, feature: FeatureSignal) -> float:
        if self.method == ImportanceMethod.PERMUTATION:
            return feature.permutation
        return feature.model

    def _is_correlated(self, feature: FeatureSignal) -> bool:
        if not self.show_correlation_groups:
            return False
        return (
            sum(
                1
                for candidate in self.preset.features
                if candidate.correlation_group == feature.correlation_group
            )
            > 1
        )

    def _method_label(self) -> str:
        if self.method == ImportanceMethod.PERMUTATION:
            return self._label("permutation importance", "permutation importance")
        return self._label("model importance", "model importance")

    def _correlation_label(self) -> str:
        correlated = sum(1 for feature in self.preset.features if self._is_correlated(feature))
        if correlated == 0:
            return self._label("none highlighted", "brak podświetlenia")
        return self._label(f"{correlated} shared-signal features", f"{correlated} powiązane cechy")

    def _diagnosis_key(self) -> str:
        top = self.top_feature()
        if top.leakage_risk and self.show_leakage_warning:
            return "leakage"
        if any(self._is_correlated(feature) for feature in self.preset.features[:3]):
            return "correlated"
        if min(feature.stability for feature in self.sorted_features()[:3]) < 0.55:
            return "unstable"
        return "stable"

    def _diagnosis_label(self) -> str:
        key = self._diagnosis_key()
        if key == "leakage":
            return self._label("possible leakage", "możliwy leakage")
        if key == "correlated":
            return self._label("shared signal", "wspólny sygnał")
        if key == "unstable":
            return self._label("unstable ranking", "niestabilny ranking")
        return self._label("stable signal", "stabilny sygnał")

    def _active_takeaway(self) -> str:
        key = self._diagnosis_key()
        if key == "leakage":
            return self._label(
                "Investigate this feature before trusting any score improvement.",
                "Sprawdź tę cechę, zanim zaufasz poprawie score.",
            )
        if key == "correlated":
            return self._label(
                "Correlated features can split credit, so read them as a group.",
                "Skorelowane cechy dzielą zasługę, więc czytaj je jako grupę.",
            )
        if key == "unstable":
            return self._label(
                "Small differences in rank are not reliable when stability is low.",
                "Małe różnice w rankingu nie są pewne przy niskiej stability.",
            )
        return self._label(
            "The top features look useful, but importance still is not causality.",
            "Top features wyglądają użytecznie, ale importance to nadal nie causality.",
        )

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


def create_feature_importance_lab_scene(context: AppContext) -> FeatureImportanceLabScene:
    """Create the unified shell Feature Importance Lab scene."""
    return FeatureImportanceLabScene(context)
