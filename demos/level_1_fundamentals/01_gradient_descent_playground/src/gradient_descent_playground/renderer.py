"""Pygame renderer for the Gradient Descent Playground demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pygame
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import NDArray

from gradient_descent_playground.challenge import ChallengeResult

type FloatArray = NDArray[np.float64]

WINDOW_WIDTH: Final[int] = 1100
WINDOW_HEIGHT: Final[int] = 720
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)

BACKGROUND_COLOR: Final[tuple[int, int, int]] = (245, 247, 250)
PANEL_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
TEXT_COLOR: Final[tuple[int, int, int]] = (30, 35, 40)
MUTED_TEXT_COLOR: Final[tuple[int, int, int]] = (100, 110, 120)
POINT_COLOR: Final[tuple[int, int, int]] = (50, 120, 220)
LINE_COLOR: Final[tuple[int, int, int]] = (220, 80, 80)
AXIS_COLOR: Final[tuple[int, int, int]] = (180, 185, 190)
LOSS_COLOR: Final[tuple[int, int, int]] = (70, 160, 90)

MAIN_RECT: Final[pygame.Rect] = pygame.Rect(40, 40, 700, 520)
SIDE_RECT: Final[pygame.Rect] = pygame.Rect(780, 40, 280, 520)
BOTTOM_RECT: Final[pygame.Rect] = pygame.Rect(40, 590, 1020, 90)

PADDING: Final[int] = 36
POINT_RADIUS: Final[int] = 5
LINE_WIDTH: Final[int] = 3
AXIS_WIDTH: Final[int] = 1
PANEL_RADIUS: Final[int] = 14
TEXT_LINE_HEIGHT: Final[int] = 28
SMALL_TEXT_LINE_HEIGHT: Final[int] = 22

MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.12


@dataclass(slots=True)
class WorldBounds:
    """Numeric coordinate bounds used to map data points onto the screen."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class GradientDescentRenderer:
    """Render Gradient Descent Playground state using Pygame."""

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
        challenge_result: ChallengeResult,
    ) -> None:
        """Draw the full frame.

        Args:
            snapshot: Current algorithm state.
            running: Whether automatic stepping is active.
            noise_std: Current synthetic data noise standard deviation.
            seed: Current synthetic dataset seed.
            challenge_result: Current challenge evaluation result.
        """
        self._screen.fill(BACKGROUND_COLOR)

        self._draw_panel(MAIN_RECT)
        self._draw_panel(SIDE_RECT)
        self._draw_panel(BOTTOM_RECT)

        self._draw_main_plot(snapshot)
        self._draw_side_panel(
            snapshot,
            running=running,
            noise_std=noise_std,
            seed=seed,
            challenge_result=challenge_result,
        )
        self._draw_bottom_panel(challenge_result)

        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw a rounded white panel."""
        pygame.draw.rect(
            self._screen,
            PANEL_COLOR,
            rect,
            border_radius=PANEL_RADIUS,
        )

    def _draw_main_plot(self, snapshot: AlgorithmSnapshot) -> None:
        """Draw data points and the current regression line."""
        features = np.asarray(snapshot.visual_state["features"], dtype=float)
        targets = np.asarray(snapshot.visual_state["targets"], dtype=float)
        predictions = np.asarray(snapshot.visual_state["predictions"], dtype=float)

        x_values = features[:, 0]
        bounds = _compute_world_bounds(x_values, targets, predictions)

        self._draw_axes(bounds)
        self._draw_points(x_values, targets, bounds)
        self._draw_regression_line(snapshot, bounds)

        self._draw_text(
            "Data world",
            MAIN_RECT.left + PADDING,
            MAIN_RECT.top + 18,
            self._font,
            TEXT_COLOR,
        )

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

    def _draw_points(
        self,
        x_values: FloatArray,
        targets: FloatArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw dataset points."""
        for x_value, target in zip(x_values, targets, strict=True):
            position = _world_to_screen(float(x_value), float(target), bounds, MAIN_RECT)
            pygame.draw.circle(self._screen, POINT_COLOR, position, POINT_RADIUS)

    def _draw_regression_line(
        self,
        snapshot: AlgorithmSnapshot,
        bounds: WorldBounds,
    ) -> None:
        """Draw the current regression line."""
        weight = float(snapshot.metrics["weight"])
        bias = float(snapshot.metrics["bias"])

        x_start = bounds.x_min
        x_end = bounds.x_max
        y_start = weight * x_start + bias
        y_end = weight * x_end + bias

        start = _world_to_screen(x_start, y_start, bounds, MAIN_RECT)
        end = _world_to_screen(x_end, y_end, bounds, MAIN_RECT)

        pygame.draw.line(self._screen, LINE_COLOR, start, end, LINE_WIDTH)

    def _draw_side_panel(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        running: bool,
        noise_std: float,
        seed: int,
        challenge_result: ChallengeResult,
    ) -> None:
        """Draw metrics, parameters, challenge, and training status."""
        x = SIDE_RECT.left + 24
        y = SIDE_RECT.top + 22

        self._draw_text("Algorithm", x, y, self._title_font, TEXT_COLOR)
        y += 46

        rows = [
            ("Status", snapshot.status),
            ("Running", "yes" if running else "no"),
            ("Step", str(snapshot.iteration)),
            ("Loss", f"{float(snapshot.metrics['loss']):.6f}"),
            ("Weight", f"{float(snapshot.metrics['weight']):.6f}"),
            ("Bias", f"{float(snapshot.metrics['bias']):.6f}"),
            ("Learning rate", f"{float(snapshot.metrics['learning_rate']):.4f}"),
            ("Noise std", f"{noise_std:.2f}"),
            ("Seed", str(seed)),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 120, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        y += 18
        self._draw_text("Loss history", x, y, self._font, TEXT_COLOR)
        y += 30

        loss_history = snapshot.visual_state["loss_history"]
        self._draw_loss_history(loss_history, pygame.Rect(x, y, 220, 110))

        y += 132
        self._draw_text("Challenge", x, y, self._font, TEXT_COLOR)
        y += 28

        challenge_rows = [
            ("Status", challenge_result.status),
            ("Target loss", f"{challenge_result.target_loss:.4f}"),
            ("Steps left", str(challenge_result.steps_remaining)),
        ]

        for label, value in challenge_rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 120, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

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

    def _draw_bottom_panel(self, challenge_result: ChallengeResult) -> None:
        """Draw keyboard controls and challenge feedback."""
        x = BOTTOM_RECT.left + 24
        y = BOTTOM_RECT.top + 18

        controls = (
            "Space: run/pause   N: step   R: reset   "
            "Up/Down: learning rate   Left/Right: noise   S: seed   Esc: quit"
        )

        self._draw_text(controls, x, y, self._small_font, TEXT_COLOR)
        self._draw_text(
            challenge_result.message,
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


def _compute_world_bounds(
    x_values: FloatArray,
    targets: FloatArray,
    predictions: FloatArray,
) -> WorldBounds:
    """Compute plot bounds with a small visual margin."""
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(np.concatenate([targets, predictions])))
    y_max = float(np.max(np.concatenate([targets, predictions])))

    x_span = max(x_max - x_min, MIN_WORLD_SPAN)
    y_span = max(y_max - y_min, MIN_WORLD_SPAN)

    x_margin = x_span * WORLD_MARGIN_RATIO
    y_margin = y_span * WORLD_MARGIN_RATIO

    return WorldBounds(
        x_min=x_min - x_margin,
        x_max=x_max + x_margin,
        y_min=y_min - y_margin,
        y_max=y_max + y_margin,
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
