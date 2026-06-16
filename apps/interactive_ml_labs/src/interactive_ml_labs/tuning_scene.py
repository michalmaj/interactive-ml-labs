"""Native Hyperparameter Tuning Lab scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
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
SECONDARY: Final[tuple[int, int, int]] = (248, 183, 96)
GOOD: Final[tuple[int, int, int]] = (147, 218, 155)
WARNING: Final[tuple[int, int, int]] = (246, 132, 134)

DEFAULT_PARAM_INDEX: Final[int] = 2


@dataclass(frozen=True, slots=True)
class TuningPoint:
    """Static scores for one hyperparameter value."""

    value: str
    train: float
    validation: float
    test: float


@dataclass(frozen=True, slots=True)
class TuningPreset:
    """Static tuning scenario."""

    name_en: str
    name_pl: str
    parameter: str
    summary_en: str
    summary_pl: str
    points: tuple[TuningPoint, TuningPoint, TuningPoint, TuningPoint, TuningPoint]

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


PRESETS: Final[tuple[TuningPreset, ...]] = (
    TuningPreset(
        name_en="k-NN neighbors",
        name_pl="k-NN neighbors",
        parameter="k",
        summary_en="Small k memorizes noise, large k smooths too much.",
        summary_pl="Małe k zapamiętuje noise, duże k zbyt mocno wygładza.",
        points=(
            TuningPoint("1", 0.99, 0.69, 0.67),
            TuningPoint("3", 0.92, 0.78, 0.77),
            TuningPoint("7", 0.86, 0.84, 0.83),
            TuningPoint("15", 0.79, 0.80, 0.79),
            TuningPoint("31", 0.70, 0.71, 0.70),
        ),
    ),
    TuningPreset(
        name_en="Tree max depth",
        name_pl="Tree max depth",
        parameter="max depth",
        summary_en="Deeper trees fit train better, but validation catches overfitting.",
        summary_pl="Głębsze drzewa lepiej fitują train, ale validation łapie overfitting.",
        points=(
            TuningPoint("1", 0.68, 0.66, 0.65),
            TuningPoint("2", 0.78, 0.76, 0.75),
            TuningPoint("4", 0.87, 0.82, 0.81),
            TuningPoint("8", 0.96, 0.74, 0.72),
            TuningPoint("None", 1.00, 0.66, 0.64),
        ),
    ),
    TuningPreset(
        name_en="Regularization strength",
        name_pl="Regularization strength",
        parameter="lambda",
        summary_en="Too much regularization underfits; too little can overfit.",
        summary_pl="Zbyt mocna regularization daje underfit; zbyt słaba może overfitować.",
        points=(
            TuningPoint("10", 0.66, 0.65, 0.64),
            TuningPoint("3", 0.74, 0.73, 0.72),
            TuningPoint("1", 0.84, 0.82, 0.81),
            TuningPoint("0.3", 0.91, 0.78, 0.77),
            TuningPoint("0", 0.98, 0.70, 0.68),
        ),
    ),
)


class HyperparameterTuningLabScene:
    """Interactive slice for validation-based hyperparameter tuning."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic tuning lab scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.param_index = DEFAULT_PARAM_INDEX

    @property
    def preset(self) -> TuningPreset:
        """Return the active tuning preset."""
        return PRESETS[self.preset_index]

    @property
    def point(self) -> TuningPoint:
        """Return the selected tuning point."""
        return self.preset.points[self.param_index]

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
        """Draw the tuning lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_curve_panel(surface, pygame.Rect(58, 132, 700, 474))
        self._draw_side_panel(surface, pygame.Rect(798, 132, 422, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
            self.param_index = DEFAULT_PARAM_INDEX
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.param_index = max(0, self.param_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.param_index = min(len(self.preset.points) - 1, self.param_index + 1)
        elif key in {pygame.K_0, pygame.K_KP0}:
            self.param_index = DEFAULT_PARAM_INDEX
        elif key == pygame.K_r:
            self.preset_index = 0
            self.param_index = DEFAULT_PARAM_INDEX

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Hyperparameter Tuning Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Pick hyperparameters by validation, not by the best train score.",
                "Wybieraj hyperparameters po validation, nie po najlepszym train score.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_curve_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Validation curve", "Validation curve"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._parameter_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        self._draw_validation_curve(surface, pygame.Rect(rect.x + 58, rect.y + 116, 584, 244))
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
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Tuning readout", "Odczyt tuningu"),
            (rect.x + 22, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        rows = (
            (self._label("preset", "preset"), self.preset.name_for_language(self._language)),
            (self._label("parameter", "parameter"), self._parameter_label()),
            (self._label("train score", "train score"), self._score_label(self.point.train)),
            (
                self._label("validation score", "validation score"),
                self._score_label(self.point.validation),
            ),
            (self._label("test score", "test score"), self._score_label(self.point.test)),
            (self._label("train-val gap", "train-val gap"), self._gap_label()),
            (self._label("best by validation", "best by validation"), self._best_value_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        y = rect.y + 68
        for label, value in rows:
            self._draw_text(surface, f"{label}: {value}", (rect.x + 22, y), self._font_small, TEXT)
            y += 24
        y += 10
        for index, preset in enumerate(PRESETS):
            color = ACCENT if index == self.preset_index else MUTED_TEXT
            self._draw_text(
                surface,
                f"{index + 1}. {preset.name_for_language(self._language)}",
                (rect.x + 22, y),
                self._font_small,
                color,
            )
            y += 24
        self._draw_wrapped(
            surface,
            self._active_takeaway(),
            (rect.x + 22, rect.bottom - 108),
            rect.width - 44,
            self._font_small,
            SECONDARY,
            line_height=17,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: choose the setting that generalizes, not the one that memorizes.",
                "Cel: wybierz ustawienie, które generalizuje, a nie zapamiętuje.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _score_label(self, value: float) -> str:
        return f"{value:.0%}"

    def _parameter_label(self) -> str:
        return f"{self.preset.parameter}={self.point.value} ({self.param_index + 1}/5)"

    def _gap(self) -> float:
        return self.point.train - self.point.validation

    def _gap_label(self) -> str:
        return f"+{self._gap():.0%}"

    def _best_index(self) -> int:
        return max(
            range(len(self.preset.points)), key=lambda index: self.preset.points[index].validation
        )

    def _best_value_label(self) -> str:
        best = self.preset.points[self._best_index()]
        return f"{self.preset.parameter}={best.value}"

    def _diagnosis_key(self) -> str:
        if self.param_index == self._best_index():
            return "candidate"
        if self._gap() >= 0.16:
            return "overfit"
        if self.point.validation < self.preset.points[self._best_index()].validation - 0.08:
            return "underfit"
        return "review"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "candidate":
            return self._label("validation candidate", "kandydat z validation")
        if diagnosis == "overfit":
            return self._label("overfit", "overfit")
        if diagnosis == "underfit":
            return self._label("underfit", "underfit")
        return self._label("review", "do sprawdzenia")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "candidate":
            return self._label(
                "Validation peaks here. Now test can estimate the chosen setup.",
                "Validation ma tu maksimum. Teraz test może ocenić wybrane ustawienie.",
            )
        if diagnosis == "overfit":
            return self._label(
                "Train score is tempting, but validation says this setting memorizes.",
                "Train score kusi, ale validation mówi, że to ustawienie zapamiętuje.",
            )
        if diagnosis == "underfit":
            return self._label(
                "Validation is still weak, so the model is too constrained.",
                "Validation nadal jest słabe, więc model jest zbyt ograniczony.",
            )
        return self._label(
            "This setting is close, but validation points to a better candidate.",
            "To ustawienie jest blisko, ale validation wskazuje lepszego kandydata.",
        )

    def _draw_validation_curve(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)
        self._draw_curve(surface, rect, tuple(point.train for point in self.preset.points), ACCENT)
        self._draw_curve(
            surface,
            rect,
            tuple(point.validation for point in self.preset.points),
            GOOD,
        )
        self._draw_curve(
            surface, rect, tuple(point.test for point in self.preset.points), SECONDARY
        )
        self._draw_selected_marker(surface, rect)
        self._draw_text(surface, "train", (rect.x, rect.bottom + 14), self._font_small, ACCENT)
        self._draw_text(
            surface, "validation", (rect.x + 80, rect.bottom + 14), self._font_small, GOOD
        )
        self._draw_text(
            surface, "test", (rect.x + 190, rect.bottom + 14), self._font_small, SECONDARY
        )

    def _draw_curve(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        scores: tuple[float, ...],
        color: tuple[int, int, int],
    ) -> None:
        points = [self._score_position(rect, index, value) for index, value in enumerate(scores)]
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 3)
        for point in points:
            pygame.draw.circle(surface, color, point, 4)

    def _draw_selected_marker(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        x = self._score_position(rect, self.param_index, self.point.validation)[0]
        pygame.draw.line(surface, WARNING, (x, rect.top), (x, rect.bottom), 2)
        self._draw_text(
            surface,
            self.point.value,
            (x - 12, rect.top + 10),
            self._font_small,
            WARNING,
        )

    def _score_position(self, rect: pygame.Rect, index: int, value: float) -> tuple[int, int]:
        x = rect.x + round(index / (len(self.preset.points) - 1) * rect.width)
        y = rect.bottom - round(max(0.0, min(1.0, value)) * rect.height)
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


def create_hyperparameter_tuning_lab_scene(context: AppContext) -> HyperparameterTuningLabScene:
    """Create the unified shell Hyperparameter Tuning Lab scene."""
    return HyperparameterTuningLabScene(context)
