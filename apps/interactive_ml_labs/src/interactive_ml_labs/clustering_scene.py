"""Native Clustering Lab scene for the unified shell."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

PLOT_RECT: Final[tuple[int, int, int, int]] = (48, 72, 820, 580)
PANEL_RECT: Final[tuple[int, int, int, int]] = (904, 72, 328, 620)
BACKGROUND: Final[tuple[int, int, int]] = (22, 25, 29)
PANEL: Final[tuple[int, int, int]] = (36, 41, 47)
PLOT_BG: Final[tuple[int, int, int]] = (18, 21, 25)
GRID: Final[tuple[int, int, int]] = (45, 51, 58)
TEXT: Final[tuple[int, int, int]] = (235, 238, 241)
MUTED_TEXT: Final[tuple[int, int, int]] = (166, 173, 181)
ACCENT: Final[tuple[int, int, int]] = (113, 204, 152)
LINK: Final[tuple[int, int, int]] = (78, 87, 97)
SPARKLINE_BG: Final[tuple[int, int, int]] = (27, 31, 36)
CENTROID_STROKE: Final[tuple[int, int, int]] = (14, 18, 22)
POINT_RADIUS: Final[int] = 5
CENTROID_RADIUS: Final[int] = 11
AUTO_STEP_SECONDS: Final[float] = 0.55
DRAG_PICK_RADIUS: Final[int] = 14
MAX_K: Final[int] = 6
MIN_K: Final[int] = 2

CLUSTER_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (119, 190, 240),
    (247, 178, 103),
    (169, 217, 113),
    (227, 121, 140),
    (188, 147, 238),
    (116, 214, 196),
)


@dataclass(frozen=True, slots=True)
class DataPreset:
    """Dataset preset available in the scene."""

    name_en: str
    name_pl: str
    key: str

    def for_language(self, language: str) -> str:
        """Return localized preset name."""
        if language == "pl":
            return self.name_pl

        return self.name_en


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in normalized plot coordinates."""

    x: float
    y: float


PRESETS: Final[tuple[DataPreset, ...]] = (
    DataPreset("Clean blobs", "Clean blobs", "blobs"),
    DataPreset("Uneven blobs", "Uneven blobs", "uneven"),
    DataPreset("Moons", "Moons", "moons"),
    DataPreset("Outliers", "Outliers", "outliers"),
)


class ClusteringLabScene:
    """Small K-Means playground used as the first real Level 3 demo."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic first Clustering Lab scene."""
        self._language = context.settings.language
        self._rng = random.Random(7)
        self._sample_seed = 7
        self.preset_index = 0
        self.k = 3
        self.iteration = 0
        self.auto_run = False
        self.show_links = False
        self.inertia = 0.0
        self.inertia_history: list[float] = []
        self.points: list[Point] = []
        self.centroids: list[Point] = []
        self.assignments: list[int] = []
        self.dragged_point_index: int | None = None
        self._auto_elapsed = 0.0
        self._font_title = make_ui_font(32, bold=True)
        self._font_heading = make_ui_font(24, bold=True)
        self._font_body = make_ui_font(19)
        self._font_small = make_ui_font(16)
        self._generate_dataset()

    @property
    def preset(self) -> DataPreset:
        """Return the active dataset preset."""
        return PRESETS[self.preset_index]

    def handle_event(self, event: object) -> SceneCommand:
        """Handle keyboard and mouse input for the playground."""
        if not isinstance(event, pygame.event.Event):
            return SceneCommand.none()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._start_drag(event.pos)
        elif event.type == pygame.MOUSEMOTION and self.dragged_point_index is not None:
            self._drag_point(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragged_point_index = None
        elif event.type != pygame.KEYDOWN:
            return SceneCommand.none()
        else:
            self._handle_keydown(event.key)

        return SceneCommand.none()

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4}:
            self.preset_index = key - pygame.K_1
            self._generate_dataset()
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self._change_k(-1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self._change_k(1)
        elif key == pygame.K_SPACE:
            self.step()
        elif key == pygame.K_a:
            self.auto_run = not self.auto_run
        elif key == pygame.K_c:
            self.show_links = not self.show_links
        elif key == pygame.K_r:
            self._reset_centroids()
        elif key == pygame.K_n:
            self._sample_seed += 1
            self._generate_dataset()

    def update(self, dt: float) -> SceneCommand:
        """Advance auto-run K-Means iterations."""
        if not self.auto_run:
            return SceneCommand.none()

        self._auto_elapsed += dt
        while self._auto_elapsed >= AUTO_STEP_SECONDS:
            self._auto_elapsed -= AUTO_STEP_SECONDS
            self.step()

        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the K-Means playground."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        plot_rect = pygame.Rect(PLOT_RECT)
        panel_rect = pygame.Rect(PANEL_RECT)
        self._draw_plot(surface, plot_rect)
        self._draw_panel(surface, panel_rect)

    def step(self) -> None:
        """Run one K-Means assignment/update iteration."""
        self._assign_points()
        next_centroids: list[Point] = []
        for cluster_index in range(self.k):
            cluster_points = [
                point
                for point, assignment in zip(self.points, self.assignments, strict=True)
                if assignment == cluster_index
            ]
            if cluster_points:
                next_centroids.append(
                    Point(
                        sum(point.x for point in cluster_points) / len(cluster_points),
                        sum(point.y for point in cluster_points) / len(cluster_points),
                    ),
                )
            else:
                next_centroids.append(self.centroids[cluster_index])

        self.centroids = next_centroids
        self._assign_points()
        self.iteration += 1
        self.inertia_history.append(self.inertia)

    def _generate_dataset(self) -> None:
        self._rng.seed(self._sample_seed + self.preset_index * 1000)
        preset_key = self.preset.key
        if preset_key == "blobs":
            self.points = self._make_blobs(
                centers=(Point(-0.58, -0.42), Point(0.08, 0.48), Point(0.58, -0.12)),
                counts=(28, 28, 28),
                spread=0.13,
            )
        elif preset_key == "uneven":
            self.points = (
                self._make_blobs((Point(-0.55, -0.36),), (42,), 0.12)
                + self._make_blobs((Point(0.42, 0.36),), (18,), 0.19)
                + self._make_blobs((Point(0.55, -0.32),), (12,), 0.09)
            )
        elif preset_key == "moons":
            self.points = self._make_moons()
        else:
            self.points = [
                *self._make_blobs(
                    centers=(Point(-0.48, -0.22), Point(0.35, 0.32), Point(0.55, -0.38)),
                    counts=(24, 24, 24),
                    spread=0.11,
                ),
                Point(-0.88, 0.76),
                Point(0.92, 0.78),
                Point(-0.82, -0.76),
            ]

        self._reset_centroids()

    def _make_blobs(
        self,
        centers: tuple[Point, ...],
        counts: tuple[int, ...],
        spread: float,
    ) -> list[Point]:
        points: list[Point] = []
        for center, count in zip(centers, counts, strict=True):
            for _ in range(count):
                points.append(
                    Point(
                        _clamp(center.x + self._rng.gauss(0.0, spread), -0.95, 0.95),
                        _clamp(center.y + self._rng.gauss(0.0, spread), -0.95, 0.95),
                    ),
                )

        return points

    def _make_moons(self) -> list[Point]:
        points: list[Point] = []
        for index in range(42):
            angle = math.pi * index / 41
            points.append(
                Point(
                    _clamp(
                        -0.12 + 0.55 * math.cos(angle) + self._rng.gauss(0.0, 0.035),
                        -0.95,
                        0.95,
                    ),
                    _clamp(
                        0.08 + 0.35 * math.sin(angle) + self._rng.gauss(0.0, 0.035),
                        -0.95,
                        0.95,
                    ),
                ),
            )
        for index in range(42):
            angle = math.pi * index / 41
            points.append(
                Point(
                    _clamp(
                        0.12 + 0.55 * math.cos(angle) + self._rng.gauss(0.0, 0.035),
                        -0.95,
                        0.95,
                    ),
                    _clamp(
                        -0.08 - 0.35 * math.sin(angle) + self._rng.gauss(0.0, 0.035),
                        -0.95,
                        0.95,
                    ),
                ),
            )

        return points

    def _reset_centroids(self) -> None:
        sample = self._rng.sample(self.points, self.k)
        self.centroids = [Point(point.x, point.y) for point in sample]
        self.iteration = 0
        self.auto_run = False
        self._auto_elapsed = 0.0
        self._assign_points()
        self.inertia_history = [self.inertia]

    def _change_k(self, delta: int) -> None:
        self.k = max(MIN_K, min(MAX_K, self.k + delta))
        self._reset_centroids()

    def _assign_points(self) -> None:
        self.assignments = [self._nearest_centroid(point) for point in self.points]
        self.inertia = sum(
            _squared_distance(point, self.centroids[assignment])
            for point, assignment in zip(self.points, self.assignments, strict=True)
        )

    def _start_drag(self, position: tuple[int, int]) -> None:
        plot_rect = pygame.Rect(PLOT_RECT)
        if not plot_rect.collidepoint(position):
            return

        nearest_index = self._nearest_point_index(position, plot_rect)
        if nearest_index is None:
            return

        self.dragged_point_index = nearest_index
        self.auto_run = False
        self._drag_point(position)

    def _drag_point(self, position: tuple[int, int]) -> None:
        if self.dragged_point_index is None:
            return

        self.points[self.dragged_point_index] = self._from_screen(position, pygame.Rect(PLOT_RECT))
        self._assign_points()
        self._replace_current_inertia_sample()

    def _nearest_point_index(
        self,
        position: tuple[int, int],
        plot_rect: pygame.Rect,
    ) -> int | None:
        nearest_index: int | None = None
        nearest_distance = DRAG_PICK_RADIUS**2
        for index, point in enumerate(self.points):
            screen_position = self._to_screen(point, plot_rect)
            distance = (screen_position[0] - position[0]) ** 2 + (
                screen_position[1] - position[1]
            ) ** 2
            if distance <= nearest_distance:
                nearest_index = index
                nearest_distance = distance

        return nearest_index

    def _nearest_centroid(self, point: Point) -> int:
        distances = [_squared_distance(point, centroid) for centroid in self.centroids]
        return min(range(len(distances)), key=distances.__getitem__)

    def _replace_current_inertia_sample(self) -> None:
        if self.inertia_history:
            self.inertia_history[-1] = self.inertia
            return

        self.inertia_history.append(self.inertia)

    def _draw_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=8)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=8)
        for offset in range(1, 4):
            x = rect.left + offset * rect.width // 4
            y = rect.top + offset * rect.height // 4
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

        if self.show_links:
            for point, assignment in zip(self.points, self.assignments, strict=True):
                pygame.draw.line(
                    surface,
                    LINK,
                    self._to_screen(point, rect),
                    self._to_screen(self.centroids[assignment], rect),
                    1,
                )

        for point, assignment in zip(self.points, self.assignments, strict=True):
            color = CLUSTER_COLORS[assignment % len(CLUSTER_COLORS)]
            pygame.draw.circle(surface, color, self._to_screen(point, rect), POINT_RADIUS)

        for index, centroid in enumerate(self.centroids):
            color = CLUSTER_COLORS[index % len(CLUSTER_COLORS)]
            position = self._to_screen(centroid, rect)
            pygame.draw.circle(surface, CENTROID_STROKE, position, CENTROID_RADIUS + 3)
            pygame.draw.circle(surface, color, position, CENTROID_RADIUS)
            pygame.draw.line(
                surface,
                CENTROID_STROKE,
                (position[0] - 8, position[1]),
                (position[0] + 8, position[1]),
                2,
            )
            pygame.draw.line(
                surface,
                CENTROID_STROKE,
                (position[0], position[1] - 8),
                (position[0], position[1] + 8),
                2,
            )

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PANEL, rect, border_radius=8)
        x = rect.x + 24
        y = rect.y + 24
        self._draw_text(surface, "Clustering Lab", (x, y), self._font_title, TEXT)
        y += 56
        self._draw_text(surface, self._label("Dataset", "Dane"), (x, y), self._font_heading, TEXT)
        y += 34
        self._draw_text(
            surface,
            self.preset.for_language(self._language),
            (x, y),
            self._font_body,
            ACCENT,
        )
        y += 38
        auto_label = self._label("on", "wł.") if self.auto_run else self._label("off", "wył.")
        links_label = self._label("on", "wł.") if self.show_links else self._label("off", "wył.")
        rows = (
            (self._label("k", "k"), str(self.k)),
            (self._label("Iteration", "Iteracja"), str(self.iteration)),
            (self._label("Inertia", "Inertia"), f"{self.inertia:.2f}"),
            (self._label("Auto-run", "Auto-run"), auto_label),
            (self._label("Links", "Linie"), links_label),
        )
        for label, value in rows:
            self._draw_text(surface, f"{label}: {value}", (x, y), self._font_body, TEXT)
            y += 28

        sparkline_rect = pygame.Rect(x, y + 4, 270, 58)
        self._draw_inertia_sparkline(surface, sparkline_rect)
        y += 82

        self._draw_text(
            surface,
            self._label("Watch", "Obserwuj"),
            (x, y),
            self._font_heading,
            TEXT,
        )
        y += 24
        y = self._draw_wrapped(
            surface,
            self._observation_hint(),
            (x, y),
            270,
            self._font_small,
            MUTED_TEXT,
            line_height=18,
        )
        y += 8
        self._draw_text(
            surface,
            self._label("Controls", "Sterowanie"),
            (x, y),
            self._font_heading,
            TEXT,
        )
        y += 26
        controls = self._control_labels()
        for index, control in enumerate(controls):
            column = index % 2
            row = index // 2
            self._draw_text(
                surface,
                control,
                (x + column * 138, y + row * 24),
                self._font_small,
                MUTED_TEXT,
            )

    def _to_screen(self, point: Point, rect: pygame.Rect) -> tuple[int, int]:
        x = rect.left + round((point.x + 1.0) * 0.5 * rect.width)
        y = rect.top + round((1.0 - (point.y + 1.0) * 0.5) * rect.height)
        return (x, y)

    def _from_screen(self, position: tuple[int, int], rect: pygame.Rect) -> Point:
        x = _clamp(((position[0] - rect.left) / rect.width) * 2.0 - 1.0, -0.95, 0.95)
        y = _clamp(((rect.bottom - position[1]) / rect.height) * 2.0 - 1.0, -0.95, 0.95)
        return Point(x, y)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        surface.blit(font.render(text, True, color), position)

    def _control_labels(self) -> tuple[str, ...]:
        return (
            f"1-4: {self._label('presets', 'presety')}",
            "- / =: k",
            f"Space: {self._label('step', 'krok')}",
            "A: auto-run",
            f"C: {self._label('links', 'linie')}",
            f"R: {self._label('reset', 'reset')}",
            f"N: {self._label('new sample', 'nowa próbka')}",
            self._label("drag points", "przesuń punkty"),
        )

    def _controls_bottom_y(self, start_y: int) -> int:
        rows = math.ceil(len(self._control_labels()) / 2)
        return start_y + rows * 24

    def _draw_inertia_sparkline(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, SPARKLINE_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_text(
            surface,
            self._label("Inertia trend", "Trend inertia"),
            (rect.x + 10, rect.y + 7),
            self._font_small,
            MUTED_TEXT,
        )
        if len(self.inertia_history) < 2:
            self._draw_text(
                surface,
                self._label("step to build history", "zrób krok, aby zobaczyć trend"),
                (rect.x + 10, rect.y + 30),
                self._font_small,
                MUTED_TEXT,
            )
            return

        values = self.inertia_history[-12:]
        minimum = min(values)
        maximum = max(values)
        span = max(maximum - minimum, 0.001)
        points: list[tuple[int, int]] = []
        chart_rect = pygame.Rect(rect.x + 10, rect.y + 24, rect.width - 20, rect.height - 34)
        for index, value in enumerate(values):
            x = chart_rect.left + round(index * chart_rect.width / max(1, len(values) - 1))
            y = chart_rect.bottom - round((value - minimum) * chart_rect.height / span)
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(surface, ACCENT, False, points, 2)
        for point in points:
            pygame.draw.circle(surface, ACCENT, point, 3)

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
    ) -> int:
        x, y = position
        for line in self._wrap_text(text, width, font):
            self._draw_text(surface, line, (x, y), font, color)
            y += line_height

        return y

    def _wrap_text(self, text: str, width: int, font: pygame.font.Font) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
                continue

            if current:
                lines.append(current)
            current = word

        if current:
            lines.append(current)

        return lines

    def _panel_controls_start_y(self, rect: pygame.Rect) -> int:
        y = rect.y + 24
        y += 56
        y += 34
        y += 38
        y += 5 * 28
        y += 82
        y += 24
        y += len(self._wrap_text(self._observation_hint(), 270, self._font_small)) * 18
        y += 8
        y += 26
        return y

    def _observation_hint(self) -> str:
        hints = {
            "blobs": self._label(
                "Clean blobs show the case where K-Means usually feels natural.",
                "Clean blobs pokazują sytuację, w której K-Means zwykle wygląda naturalnie.",
            ),
            "uneven": self._label(
                "Uneven blobs reveal how cluster size and density can pull centroids.",
                "Uneven blobs pokazują, jak rozmiar i gęstość klastrów przesuwają centroidy.",
            ),
            "moons": self._label(
                "Moons make the round-cluster assumption visible.",
                "Moons dobrze pokazują założenie o okrągłych klastrach.",
            ),
            "outliers": self._label(
                "Outliers show how a few distant points can distort centroid placement.",
                "Outliery pokazują, jak kilka dalekich punktów zniekształca położenie centroidów.",
            ),
        }
        return hints[self.preset.key]

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl

        return en


def _squared_distance(left: Point, right: Point) -> float:
    return (left.x - right.x) ** 2 + (left.y - right.y) ** 2


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def create_clustering_lab_scene(context: AppContext) -> ClusteringLabScene:
    """Create the unified shell Clustering Lab scene."""
    return ClusteringLabScene(context)
