"""Pygame renderer for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pygame
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import NDArray

from logistic_regression_boundary_lab.probability_grid import ProbabilityGrid

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]

WINDOW_WIDTH: Final[int] = 1100
WINDOW_HEIGHT: Final[int] = 720
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)

BACKGROUND_COLOR: Final[tuple[int, int, int]] = (245, 247, 250)
PANEL_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
TEXT_COLOR: Final[tuple[int, int, int]] = (30, 35, 40)
MUTED_TEXT_COLOR: Final[tuple[int, int, int]] = (100, 110, 120)
CLASS_ZERO_COLOR: Final[tuple[int, int, int]] = (60, 130, 230)
CLASS_ONE_COLOR: Final[tuple[int, int, int]] = (235, 130, 60)
PROBABILITY_LOW_COLOR: Final[tuple[int, int, int]] = (219, 233, 252)
PROBABILITY_HIGH_COLOR: Final[tuple[int, int, int]] = (252, 230, 215)
BOUNDARY_COLOR: Final[tuple[int, int, int]] = (40, 40, 40)
AXIS_COLOR: Final[tuple[int, int, int]] = (180, 185, 190)
LOSS_COLOR: Final[tuple[int, int, int]] = (70, 160, 90)
ERROR_COLOR: Final[tuple[int, int, int]] = (190, 60, 60)

MAIN_RECT: Final[pygame.Rect] = pygame.Rect(40, 40, 700, 520)
SIDE_RECT: Final[pygame.Rect] = pygame.Rect(780, 40, 280, 520)
BOTTOM_RECT: Final[pygame.Rect] = pygame.Rect(40, 590, 1020, 90)

PADDING: Final[int] = 36
POINT_RADIUS: Final[int] = 6
AXIS_WIDTH: Final[int] = 1
BOUNDARY_WIDTH: Final[int] = 3
ERROR_MARK_WIDTH: Final[int] = 2
PANEL_RADIUS: Final[int] = 14
TEXT_LINE_HEIGHT: Final[int] = 28
SMALL_TEXT_LINE_HEIGHT: Final[int] = 22

MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.18
MIN_BOUNDARY_WEIGHT_NORM: Final[float] = 1.0e-8

CLASS_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    0: CLASS_ZERO_COLOR,
    1: CLASS_ONE_COLOR,
}


@dataclass(slots=True)
class WorldBounds:
    """Numeric coordinate bounds used to map data points onto the screen."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class LogisticRegressionRenderer:
    """Render logistic regression state using Pygame."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initialize renderer resources.

        Args:
            screen: Target Pygame surface.
        """
        self._screen = screen
        self._font = pygame.font.Font(None, 28)
        self._small_font = pygame.font.Font(None, 22)
        self._title_font = pygame.font.Font(None, 36)

    def draw(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        running: bool,
        noise_std: float,
        seed: int,
        probability_grid: ProbabilityGrid,
    ) -> None:
        """Draw the full frame.

        Args:
            snapshot: Current algorithm state.
            running: Whether automatic training is active.
            noise_std: Current synthetic data noise standard deviation.
            seed: Current synthetic dataset seed.
            probability_grid: Positive-class probabilities over the background grid.
        """
        self._screen.fill(BACKGROUND_COLOR)

        self._draw_panel(MAIN_RECT)
        self._draw_panel(SIDE_RECT)
        self._draw_panel(BOTTOM_RECT)

        self._draw_main_plot(snapshot, probability_grid)
        self._draw_side_panel(snapshot, running=running, noise_std=noise_std, seed=seed)
        self._draw_bottom_panel(snapshot)

        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw a rounded white panel."""
        pygame.draw.rect(
            self._screen,
            PANEL_COLOR,
            rect,
            border_radius=PANEL_RADIUS,
        )

    def _draw_main_plot(
        self,
        snapshot: AlgorithmSnapshot,
        probability_grid: ProbabilityGrid,
    ) -> None:
        """Draw probability background, data points, and decision boundary."""
        features = np.asarray(snapshot.visual_state["features"], dtype=float)
        targets = np.asarray(snapshot.visual_state["targets"], dtype=int)
        predictions = np.asarray(snapshot.visual_state["predictions"], dtype=int)
        weights = np.asarray(snapshot.visual_state["weights"], dtype=float)
        bias = float(snapshot.visual_state["bias"])
        threshold = float(snapshot.visual_state["threshold"])

        bounds = _bounds_from_probability_grid(probability_grid)

        self._draw_probability_background(probability_grid, bounds)
        self._draw_axes(bounds)
        self._draw_decision_boundary(
            weights=weights,
            bias=bias,
            threshold=threshold,
            bounds=bounds,
        )
        self._draw_points(features, targets, predictions, bounds)

        self._draw_text(
            "Probability background",
            MAIN_RECT.left + PADDING,
            MAIN_RECT.top + 18,
            self._font,
            TEXT_COLOR,
        )

    def _draw_probability_background(
        self,
        probability_grid: ProbabilityGrid,
        bounds: WorldBounds,
    ) -> None:
        """Draw positive-class probability as a soft colored background."""
        x_values = probability_grid.x_values
        y_values = probability_grid.y_values
        probabilities = probability_grid.probabilities

        x_step = _grid_step(x_values)
        y_step = _grid_step(y_values)

        for y_index, y_value in enumerate(y_values):
            for x_index, x_value in enumerate(x_values):
                probability = float(probabilities[y_index, x_index])
                color = _probability_to_color(probability)

                top_left = _world_to_screen(
                    float(x_value - x_step / 2.0),
                    float(y_value + y_step / 2.0),
                    bounds,
                    MAIN_RECT,
                )
                bottom_right = _world_to_screen(
                    float(x_value + x_step / 2.0),
                    float(y_value - y_step / 2.0),
                    bounds,
                    MAIN_RECT,
                )

                rect = pygame.Rect(
                    top_left[0],
                    top_left[1],
                    max(1, bottom_right[0] - top_left[0] + 1),
                    max(1, bottom_right[1] - top_left[1] + 1),
                )
                pygame.draw.rect(self._screen, color, rect)

    def _draw_axes(self, bounds: WorldBounds) -> None:
        """Draw x=0 and y=0 axes when visible."""
        if bounds.y_min <= 0.0 <= bounds.y_max:
            start = _world_to_screen(bounds.x_min, 0.0, bounds, MAIN_RECT)
            end = _world_to_screen(bounds.x_max, 0.0, bounds, MAIN_RECT)
            pygame.draw.line(self._screen, AXIS_COLOR, start, end, AXIS_WIDTH)

        if bounds.x_min <= 0.0 <= bounds.x_max:
            start = _world_to_screen(0.0, bounds.y_min, bounds, MAIN_RECT)
            end = _world_to_screen(0.0, bounds.y_max, bounds, MAIN_RECT)
            pygame.draw.line(self._screen, AXIS_COLOR, start, end, AXIS_WIDTH)

    def _draw_decision_boundary(
        self,
        *,
        weights: FloatArray,
        bias: float,
        threshold: float,
        bounds: WorldBounds,
    ) -> None:
        """Draw the logistic regression decision boundary."""
        if threshold <= 0.0 or threshold >= 1.0:
            return

        weight_1 = float(weights[0])
        weight_2 = float(weights[1])
        weight_norm = float(np.hypot(weight_1, weight_2))

        if weight_norm <= MIN_BOUNDARY_WEIGHT_NORM:
            return

        target_score = _logit(threshold)

        if abs(weight_2) > MIN_BOUNDARY_WEIGHT_NORM:
            x_start = bounds.x_min
            x_end = bounds.x_max
            y_start = (target_score - bias - weight_1 * x_start) / weight_2
            y_end = (target_score - bias - weight_1 * x_end) / weight_2

            start = _world_to_screen(x_start, y_start, bounds, MAIN_RECT)
            end = _world_to_screen(x_end, y_end, bounds, MAIN_RECT)
        else:
            x_value = (target_score - bias) / weight_1
            start = _world_to_screen(x_value, bounds.y_min, bounds, MAIN_RECT)
            end = _world_to_screen(x_value, bounds.y_max, bounds, MAIN_RECT)

        pygame.draw.line(self._screen, BOUNDARY_COLOR, start, end, BOUNDARY_WIDTH)

    def _draw_points(
        self,
        features: FloatArray,
        targets: IntArray,
        predictions: IntArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw labeled points and mark misclassified examples."""
        for point, target, prediction in zip(features, targets, predictions, strict=True):
            position = _world_to_screen(float(point[0]), float(point[1]), bounds, MAIN_RECT)
            color = CLASS_COLORS.get(int(target), MUTED_TEXT_COLOR)

            pygame.draw.circle(self._screen, color, position, POINT_RADIUS)

            if int(target) != int(prediction):
                self._draw_error_mark(position)

    def _draw_error_mark(self, position: tuple[int, int]) -> None:
        """Draw a small X mark over a misclassified point."""
        x, y = position
        offset = POINT_RADIUS + 2

        pygame.draw.line(
            self._screen,
            ERROR_COLOR,
            (x - offset, y - offset),
            (x + offset, y + offset),
            ERROR_MARK_WIDTH,
        )
        pygame.draw.line(
            self._screen,
            ERROR_COLOR,
            (x - offset, y + offset),
            (x + offset, y - offset),
            ERROR_MARK_WIDTH,
        )

    def _draw_side_panel(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        running: bool,
        noise_std: float,
        seed: int,
    ) -> None:
        """Draw metrics, parameters, and loss history."""
        x = SIDE_RECT.left + 24
        y = SIDE_RECT.top + 22

        self._draw_text("Logistic regression", x, y, self._title_font, TEXT_COLOR)
        y += 46

        rows = [
            ("Status", snapshot.status),
            ("Running", "yes" if running else "no"),
            ("Step", str(snapshot.iteration)),
            ("Loss", f"{float(snapshot.metrics['loss']):.6f}"),
            ("Accuracy", f"{float(snapshot.metrics['accuracy']):.3f}"),
            ("Precision", f"{float(snapshot.metrics['precision']):.3f}"),
            ("Recall", f"{float(snapshot.metrics['recall']):.3f}"),
            ("Learning rate", f"{float(snapshot.metrics['learning_rate']):.3f}"),
            ("Threshold", f"{float(snapshot.metrics['threshold']):.2f}"),
            ("Noise std", f"{noise_std:.2f}"),
            ("Seed", str(seed)),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 124, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        y += 12
        self._draw_text("Loss history", x, y, self._font, TEXT_COLOR)
        y += 28

        loss_history = tuple(float(value) for value in snapshot.visual_state["loss_history"])
        self._draw_loss_history(loss_history, pygame.Rect(x, y, 220, 92))

    def _draw_loss_history(
        self,
        loss_history: tuple[float, ...],
        rect: pygame.Rect,
    ) -> None:
        """Draw a compact loss history chart."""
        pygame.draw.rect(self._screen, BACKGROUND_COLOR, rect, border_radius=8)

        if len(loss_history) < 2:
            return

        max_loss = max(loss_history)
        min_loss = min(loss_history)
        loss_span = max(max_loss - min_loss, MIN_WORLD_SPAN)

        points: list[tuple[int, int]] = []
        last_index = len(loss_history) - 1

        for index, loss in enumerate(loss_history):
            x = rect.left + int((index / last_index) * rect.width)
            normalized = (loss - min_loss) / loss_span
            y = rect.bottom - int(normalized * rect.height)
            points.append((x, y))

        pygame.draw.lines(self._screen, LOSS_COLOR, False, points, 2)

    def _draw_bottom_panel(self, snapshot: AlgorithmSnapshot) -> None:
        """Draw keyboard controls and short explanation."""
        x = BOTTOM_RECT.left + 24
        y = BOTTOM_RECT.top + 14

        controls = (
            "Space: run/pause   N: step   R: reset   Up/Down: learning rate   "
            "Q/E: threshold   Left/Right: noise   S: seed   Esc: quit"
        )

        self._draw_text(controls, x, y, self._small_font, TEXT_COLOR)

        explanation = _build_explanation(snapshot)
        self._draw_text(
            explanation,
            x,
            y + TEXT_LINE_HEIGHT,
            self._small_font,
            MUTED_TEXT_COLOR,
        )

    def _draw_text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        """Draw text at a fixed screen position."""
        surface = font.render(text, True, color)
        self._screen.blit(surface, (x, y))


def _build_explanation(snapshot: AlgorithmSnapshot) -> str:
    """Build a short explanation for the current model state."""
    if snapshot.iteration == 0:
        return "The background shows probability of class_1. Press N or Space to train."

    loss = float(snapshot.metrics["loss"])
    accuracy = float(snapshot.metrics["accuracy"])
    threshold = float(snapshot.metrics["threshold"])

    return f"Probability background changes as weights learn. Loss: {loss:.4f}, \
        accuracy: {accuracy:.2f}, threshold: {threshold:.2f}."


def _bounds_from_probability_grid(probability_grid: ProbabilityGrid) -> WorldBounds:
    """Build world bounds from a probability grid."""
    return WorldBounds(
        x_min=float(np.min(probability_grid.x_values)),
        x_max=float(np.max(probability_grid.x_values)),
        y_min=float(np.min(probability_grid.y_values)),
        y_max=float(np.max(probability_grid.y_values)),
    )


def _world_to_screen(
    x_value: float,
    y_value: float,
    bounds: WorldBounds,
    rect: pygame.Rect,
) -> tuple[int, int]:
    """Convert numeric world coordinates into screen coordinates."""
    drawable_width = rect.width - 2 * PADDING
    drawable_height = rect.height - 2 * PADDING

    x_ratio = (x_value - bounds.x_min) / (bounds.x_max - bounds.x_min)
    y_ratio = (y_value - bounds.y_min) / (bounds.y_max - bounds.y_min)

    screen_x = rect.left + PADDING + int(x_ratio * drawable_width)
    screen_y = rect.bottom - PADDING - int(y_ratio * drawable_height)

    return screen_x, screen_y


def _logit(probability: float) -> float:
    """Convert probability into log-odds."""
    return float(np.log(probability / (1.0 - probability)))


def _grid_step(values: FloatArray) -> float:
    """Return spacing between grid values."""
    if len(values) < 2:
        return MIN_WORLD_SPAN

    return float(values[1] - values[0])


def _probability_to_color(probability: float) -> tuple[int, int, int]:
    """Convert probability into an interpolated background color."""
    clipped_probability = min(1.0, max(0.0, probability))

    return tuple(
        round(low + (high - low) * clipped_probability)
        for low, high in zip(PROBABILITY_LOW_COLOR, PROBABILITY_HIGH_COLOR, strict=True)
    )
