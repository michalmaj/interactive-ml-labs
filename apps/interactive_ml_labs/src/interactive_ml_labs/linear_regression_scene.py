"""Native Linear Regression Line Fit Lab scene for the unified shell."""

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

SLOPE_STEP: Final[float] = 0.2
INTERCEPT_STEP: Final[float] = 0.2
LINEAR_REGRESSION_LESSON_ID: Final[str] = "error_linear_regression_line_fit"
BALANCE_RESIDUALS_TASK_ID: Final[str] = "balance_residuals"
COMPARE_LEAST_SQUARES_TASK_ID: Final[str] = "compare_least_squares"


@dataclass(frozen=True, slots=True)
class RegressionPreset:
    """Static regression dataset."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    points: tuple[tuple[float, float], ...]
    start_slope: float
    start_intercept: float

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


PRESETS: Final[tuple[RegressionPreset, ...]] = (
    RegressionPreset(
        name_en="Clean trend",
        name_pl="Czysty trend",
        summary_en="Most points follow one clear linear relationship.",
        summary_pl="Większość punktów układa się wokół jednej czytelnej prostej.",
        points=(
            (-3.0, -2.2),
            (-2.0, -1.3),
            (-1.0, -0.4),
            (0.0, 0.4),
            (1.0, 1.6),
            (2.0, 2.3),
            (3.0, 3.2),
        ),
        start_slope=0.2,
        start_intercept=0.0,
    ),
    RegressionPreset(
        name_en="Noisy trend",
        name_pl="Szum wokół trendu",
        summary_en="Noise makes the best line useful, but never perfect.",
        summary_pl="Noise sprawia, że najlepsza prosta pomaga, ale nie będzie idealna.",
        points=(
            (-3.0, -1.4),
            (-2.0, -2.2),
            (-1.0, -0.2),
            (0.0, -0.1),
            (1.0, 1.8),
            (2.0, 1.1),
            (3.0, 3.4),
        ),
        start_slope=0.0,
        start_intercept=0.8,
    ),
    RegressionPreset(
        name_en="Offset target",
        name_pl="Przesunięty target",
        summary_en="The slope is close, but the intercept needs to move the line up.",
        summary_pl="Slope jest blisko, ale intercept musi przesunąć linię w górę.",
        points=(
            (-3.0, -0.9),
            (-2.0, -0.3),
            (-1.0, 0.6),
            (0.0, 1.1),
            (1.0, 1.8),
            (2.0, 2.8),
            (3.0, 3.2),
        ),
        start_slope=0.7,
        start_intercept=-0.8,
    ),
)


class LinearRegressionLineFitLabScene:
    """Interactive slice for fitting a simple linear regression line."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create the deterministic line fitting scene."""
        self._context = context
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.slope = PRESETS[0].start_slope
        self.intercept = PRESETS[0].start_intercept
        self._manual_slope_adjusted = False
        self._manual_intercept_adjusted = False

    @property
    def preset(self) -> RegressionPreset:
        """Return the active dataset preset."""
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
        """Draw the regression lab."""
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
            self._reset_parameters()
        elif key == pygame.K_LEFT:
            self.slope -= SLOPE_STEP
            self._manual_slope_adjusted = True
            self._record_manual_fit_progress()
        elif key == pygame.K_RIGHT:
            self.slope += SLOPE_STEP
            self._manual_slope_adjusted = True
            self._record_manual_fit_progress()
        elif key == pygame.K_DOWN:
            self.intercept -= INTERCEPT_STEP
            self._manual_intercept_adjusted = True
            self._record_manual_fit_progress()
        elif key == pygame.K_UP:
            self.intercept += INTERCEPT_STEP
            self._manual_intercept_adjusted = True
            self._record_manual_fit_progress()
        elif key == pygame.K_f:
            self.slope, self.intercept = self._best_fit()
            self._record_least_squares_progress()
        elif key == pygame.K_r:
            self.preset_index = 0
            self._reset_parameters()

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Linear Regression Line Fit Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Move a line and watch residuals turn into loss.",
                "Przesuwaj prostą i zobacz, jak residuals zamieniają się w loss.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_plot_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Line and residuals", "Prosta i residuals"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._equation_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            SECONDARY,
        )
        plot_rect = pygame.Rect(rect.x + 70, rect.y + 104, 560, 270)
        self._draw_regression_plot(surface, plot_rect)
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
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("slope", "slope"), f"{self.slope:+.1f}"),
            (self._label("intercept", "intercept"), f"{self.intercept:+.1f}"),
            (self._label("MSE loss", "MSE loss"), f"{self._mse():.2f}"),
            (self._label("best slope", "najlepszy slope"), f"{self._best_fit()[0]:+.1f}"),
            (self._label("best intercept", "najlepszy intercept"), f"{self._best_fit()[1]:+.1f}"),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("Fit readout", "Odczyt dopasowania"),
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
                "Goal: fit a useful line by balancing all residuals, not one point.",
                "Cel: dopasuj użyteczną prostą, równoważąc wszystkie residuals.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_regression_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_grid(surface, rect)
        self._draw_residuals(surface, rect)
        self._draw_line(surface, rect, self.slope, self.intercept, ACCENT, width=3)
        best_slope, best_intercept = self._best_fit()
        self._draw_line(surface, rect, best_slope, best_intercept, GOOD, width=2)
        for point in self.preset.points:
            pygame.draw.circle(surface, TEXT, self._to_screen(rect, point[0], point[1]), 5)
        self._draw_text(surface, "current", (rect.x, rect.bottom + 14), self._font_small, ACCENT)
        self._draw_text(
            surface, "least squares", (rect.x + 92, rect.bottom + 14), self._font_small, GOOD
        )

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_residuals(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for x_value, y_value in self.preset.points:
            predicted = self._predict(x_value)
            start = self._to_screen(rect, x_value, y_value)
            end = self._to_screen(rect, x_value, predicted)
            pygame.draw.line(surface, WARNING, start, end, 2)

    def _draw_line(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        slope: float,
        intercept: float,
        color: tuple[int, int, int],
        *,
        width: int,
    ) -> None:
        left = self._to_screen(rect, -3.4, (slope * -3.4) + intercept)
        right = self._to_screen(rect, 3.4, (slope * 3.4) + intercept)
        pygame.draw.line(surface, color, left, right, width)

    def _to_screen(self, rect: pygame.Rect, x_value: float, y_value: float) -> tuple[int, int]:
        x = rect.x + round((x_value + 3.5) / 7.0 * rect.width)
        y = rect.bottom - round((y_value + 3.5) / 7.0 * rect.height)
        return (x, y)

    def _reset_parameters(self) -> None:
        self.slope = self.preset.start_slope
        self.intercept = self.preset.start_intercept
        self._manual_slope_adjusted = False
        self._manual_intercept_adjusted = False

    def _predict(self, x_value: float) -> float:
        return (self.slope * x_value) + self.intercept

    def _mse(self) -> float:
        squared_errors = tuple((y - self._predict(x)) ** 2 for x, y in self.preset.points)
        return sum(squared_errors) / len(squared_errors)

    def _best_fit(self) -> tuple[float, float]:
        points = self.preset.points
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        slope = numerator / denominator
        intercept = mean_y - (slope * mean_x)
        return (slope, intercept)

    def _best_mse(self) -> float:
        slope, intercept = self._best_fit()
        squared_errors = tuple((y - ((slope * x) + intercept)) ** 2 for x, y in self.preset.points)
        return sum(squared_errors) / len(squared_errors)

    def _diagnosis_key(self) -> str:
        if self._mse() <= self._best_mse() + 0.08:
            return "close"
        if abs(self.slope - self._best_fit()[0]) > abs(self.intercept - self._best_fit()[1]):
            return "slope"
        return "intercept"

    def _diagnosis_label(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "close":
            return self._label("near best fit", "blisko best fit")
        if diagnosis == "slope":
            return self._label("adjust slope", "popraw slope")
        return self._label("adjust intercept", "popraw intercept")

    def _active_takeaway(self) -> str:
        diagnosis = self._diagnosis_key()
        if diagnosis == "close":
            return self._label(
                "Residuals are balanced. A few points can still miss the line.",
                "Residuals są zbalansowane. Pojedyncze punkty nadal mogą nie trafiać w linię.",
            )
        if diagnosis == "slope":
            return self._label(
                "The line tilts the wrong way, so errors grow across the x-axis.",
                "Prosta jest źle nachylona, więc błędy rosną wzdłuż osi x.",
            )
        return self._label(
            "The line is shifted too high or too low for most points.",
            "Prosta jest przesunięta za wysoko albo za nisko względem większości punktów.",
        )

    def _record_manual_fit_progress(self) -> None:
        """Complete the manual residual-balancing task after both parameters move."""
        if self._context.selected_lesson_id != LINEAR_REGRESSION_LESSON_ID:
            return
        if not (self._manual_slope_adjusted and self._manual_intercept_adjusted):
            return

        self._context.progress.complete_task(
            LINEAR_REGRESSION_LESSON_ID,
            BALANCE_RESIDUALS_TASK_ID,
        )
        self._mark_lesson_completed_if_ready()

    def _record_least_squares_progress(self) -> None:
        """Complete the least-squares comparison task."""
        if self._context.selected_lesson_id != LINEAR_REGRESSION_LESSON_ID:
            return

        self._context.progress.complete_task(
            LINEAR_REGRESSION_LESSON_ID,
            COMPARE_LEAST_SQUARES_TASK_ID,
        )
        self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both tasks are done."""
        progress = self._context.progress.lessons.get(LINEAR_REGRESSION_LESSON_ID)
        if progress is None:
            return

        required_tasks = {BALANCE_RESIDUALS_TASK_ID, COMPARE_LEAST_SQUARES_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(LINEAR_REGRESSION_LESSON_ID)

    def _equation_label(self) -> str:
        return f"y = {self.slope:+.1f}x {self.intercept:+.1f}"

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


def create_linear_regression_line_fit_lab_scene(
    context: AppContext,
) -> LinearRegressionLineFitLabScene:
    """Create the unified shell Linear Regression Line Fit Lab scene."""
    return LinearRegressionLineFitLabScene(context)
