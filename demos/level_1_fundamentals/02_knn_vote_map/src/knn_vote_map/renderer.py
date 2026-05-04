"""Pygame renderer for the k-NN Vote Map demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pygame
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import NDArray

from knn_vote_map.classifier import Neighbor

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
QUERY_COLOR: Final[tuple[int, int, int]] = (40, 40, 40)
NEIGHBOR_LINE_COLOR: Final[tuple[int, int, int]] = (90, 90, 90)
AXIS_COLOR: Final[tuple[int, int, int]] = (180, 185, 190)

MAIN_RECT: Final[pygame.Rect] = pygame.Rect(40, 40, 700, 520)
SIDE_RECT: Final[pygame.Rect] = pygame.Rect(780, 40, 280, 520)
BOTTOM_RECT: Final[pygame.Rect] = pygame.Rect(40, 590, 1020, 90)

PADDING: Final[int] = 36
POINT_RADIUS: Final[int] = 6
QUERY_RADIUS: Final[int] = 9
NEIGHBOR_RADIUS: Final[int] = 9
AXIS_WIDTH: Final[int] = 1
NEIGHBOR_LINE_WIDTH: Final[int] = 2
PANEL_RADIUS: Final[int] = 14
TEXT_LINE_HEIGHT: Final[int] = 28
SMALL_TEXT_LINE_HEIGHT: Final[int] = 22

MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.18

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


class KNNVoteMapRenderer:
    """Render k-NN classification state using Pygame."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initialize renderer resources.

        Args:
            screen: Target Pygame surface.
        """
        self._screen = screen
        self._font = pygame.font.Font(None, 28)
        self._small_font = pygame.font.Font(None, 22)
        self._title_font = pygame.font.Font(None, 36)
        self._last_bounds: WorldBounds | None = None

    def draw(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        noise_std: float,
        seed: int,
    ) -> None:
        """Draw the full frame.

        Args:
            snapshot: Current classifier state.
            noise_std: Current synthetic data noise standard deviation.
            seed: Current synthetic dataset seed.
        """
        self._screen.fill(BACKGROUND_COLOR)

        self._draw_panel(MAIN_RECT)
        self._draw_panel(SIDE_RECT)
        self._draw_panel(BOTTOM_RECT)

        self._draw_main_plot(snapshot)
        self._draw_side_panel(snapshot, noise_std=noise_std, seed=seed)
        self._draw_bottom_panel(snapshot)

        pygame.display.flip()

    def screen_to_world(self, position: tuple[int, int]) -> tuple[float, float] | None:
        """Convert a screen position into world coordinates.

        Args:
            position: Mouse position in screen coordinates.

        Returns:
            World coordinate pair if the position is inside the plot area,
            otherwise `None`.
        """
        if self._last_bounds is None:
            return None

        if not _drawable_rect(MAIN_RECT).collidepoint(position):
            return None

        return _screen_to_world(position, self._last_bounds, MAIN_RECT)

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw a rounded white panel."""
        pygame.draw.rect(
            self._screen,
            PANEL_COLOR,
            rect,
            border_radius=PANEL_RADIUS,
        )

    def _draw_main_plot(self, snapshot: AlgorithmSnapshot) -> None:
        """Draw training points, query point, and nearest neighbors."""
        features = np.asarray(snapshot.visual_state["features"], dtype=float)
        targets = np.asarray(snapshot.visual_state["targets"], dtype=int)

        query_point = snapshot.visual_state.get("query_point")
        query_array = None if query_point is None else np.asarray(query_point, dtype=float)

        bounds = _compute_world_bounds(features, query_array)
        self._last_bounds = bounds

        self._draw_axes(bounds)
        self._draw_training_points(features, targets, bounds)

        neighbors = snapshot.visual_state.get("neighbors", ())
        if query_array is not None:
            self._draw_neighbor_links(query_array, neighbors, features, bounds)
            self._draw_query_point(query_array, bounds)

        self._draw_text(
            "Vote map",
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

    def _draw_training_points(
        self,
        features: FloatArray,
        targets: IntArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw labeled training points."""
        for point, label in zip(features, targets, strict=True):
            color = CLASS_COLORS.get(int(label), MUTED_TEXT_COLOR)
            position = _world_to_screen(float(point[0]), float(point[1]), bounds, MAIN_RECT)
            pygame.draw.circle(self._screen, color, position, POINT_RADIUS)

    def _draw_neighbor_links(
        self,
        query_point: FloatArray,
        neighbors: object,
        features: FloatArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw links from query point to selected neighbors."""
        query_position = _world_to_screen(
            float(query_point[0]),
            float(query_point[1]),
            bounds,
            MAIN_RECT,
        )

        for neighbor in neighbors:
            if not isinstance(neighbor, Neighbor):
                continue

            neighbor_point = features[neighbor.index]
            neighbor_position = _world_to_screen(
                float(neighbor_point[0]),
                float(neighbor_point[1]),
                bounds,
                MAIN_RECT,
            )

            pygame.draw.line(
                self._screen,
                NEIGHBOR_LINE_COLOR,
                query_position,
                neighbor_position,
                NEIGHBOR_LINE_WIDTH,
            )
            pygame.draw.circle(
                self._screen,
                CLASS_COLORS.get(neighbor.label, MUTED_TEXT_COLOR),
                neighbor_position,
                NEIGHBOR_RADIUS,
                width=2,
            )

    def _draw_query_point(self, query_point: FloatArray, bounds: WorldBounds) -> None:
        """Draw the current query point."""
        position = _world_to_screen(
            float(query_point[0]),
            float(query_point[1]),
            bounds,
            MAIN_RECT,
        )
        pygame.draw.circle(self._screen, QUERY_COLOR, position, QUERY_RADIUS)
        pygame.draw.circle(self._screen, PANEL_COLOR, position, QUERY_RADIUS - 3)

    def _draw_side_panel(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        noise_std: float,
        seed: int,
    ) -> None:
        """Draw metrics, parameters, and prediction details."""
        x = SIDE_RECT.left + 24
        y = SIDE_RECT.top + 22

        self._draw_text("k-NN classifier", x, y, self._title_font, TEXT_COLOR)
        y += 46

        rows = [
            ("Status", snapshot.status),
            ("k", str(snapshot.metrics["k"])),
            ("Samples", str(snapshot.metrics["sample_count"])),
            ("Classes", str(snapshot.metrics["class_count"])),
            ("Noise std", f"{noise_std:.2f}"),
            ("Seed", str(seed)),
        ]

        if snapshot.metrics.get("has_prediction"):
            rows.append(("Prediction", str(snapshot.metrics["predicted_label"])))

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 120, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        y += 18
        self._draw_text("Votes", x, y, self._font, TEXT_COLOR)
        y += 30

        vote_counts = snapshot.visual_state.get("vote_counts", {})
        if isinstance(vote_counts, dict) and vote_counts:
            for label, count in sorted(vote_counts.items()):
                self._draw_text(
                    f"class_{label}:",
                    x,
                    y,
                    self._small_font,
                    MUTED_TEXT_COLOR,
                )
                self._draw_text(str(count), x + 120, y, self._small_font, TEXT_COLOR)
                y += SMALL_TEXT_LINE_HEIGHT
        else:
            self._draw_text("No query point yet.", x, y, self._small_font, MUTED_TEXT_COLOR)

        y += 24
        self._draw_text("Legend", x, y, self._font, TEXT_COLOR)
        y += 30

        self._draw_legend_item(x, y, CLASS_ZERO_COLOR, "class_0")
        y += SMALL_TEXT_LINE_HEIGHT
        self._draw_legend_item(x, y, CLASS_ONE_COLOR, "class_1")
        y += SMALL_TEXT_LINE_HEIGHT
        self._draw_legend_item(x, y, QUERY_COLOR, "query point")

    def _draw_legend_item(
        self,
        x: int,
        y: int,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        """Draw one legend item."""
        pygame.draw.circle(self._screen, color, (x + 8, y + 8), 6)
        self._draw_text(label, x + 24, y, self._small_font, TEXT_COLOR)

    def _draw_bottom_panel(self, snapshot: AlgorithmSnapshot) -> None:
        """Draw keyboard and mouse controls with a short explanation."""
        x = BOTTOM_RECT.left + 24
        y = BOTTOM_RECT.top + 14

        controls = (
            "Click map: classify point   N: random query   R: reset   Up/Down: k   "
            "Left/Right: noise   S: seed   Esc: quit"
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
    """Build a short explanation line for the current state."""
    if not snapshot.metrics.get("has_prediction"):
        return "Click on the map or press N to classify a query point."

    prediction = snapshot.metrics["predicted_label"]
    k = snapshot.metrics["k"]

    return f"The query point was classified as class_{prediction} by voting among {k} neighbors."


def _compute_world_bounds(
    features: FloatArray,
    query_point: FloatArray | None,
) -> WorldBounds:
    """Compute plot bounds with a small visual margin."""
    points = features

    if query_point is not None:
        points = np.vstack([features, query_point.reshape(1, -1)])

    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))

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


def _screen_to_world(
    position: tuple[int, int],
    bounds: WorldBounds,
    rect: pygame.Rect,
) -> tuple[float, float]:
    """Convert screen coordinates into numeric world coordinates."""
    drawable = _drawable_rect(rect)

    x_ratio = (position[0] - drawable.left) / drawable.width
    y_ratio = (drawable.bottom - position[1]) / drawable.height

    world_x = bounds.x_min + x_ratio * (bounds.x_max - bounds.x_min)
    world_y = bounds.y_min + y_ratio * (bounds.y_max - bounds.y_min)

    return world_x, world_y


def _drawable_rect(rect: pygame.Rect) -> pygame.Rect:
    """Return the inner drawable area of a plot panel."""
    return pygame.Rect(
        rect.left + PADDING,
        rect.top + PADDING,
        rect.width - 2 * PADDING,
        rect.height - 2 * PADDING,
    )
