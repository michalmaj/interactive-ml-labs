"""Native K-Means Intro Lab scene for the unified shell."""

from __future__ import annotations

import random
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

BACKGROUND: Final[tuple[int, int, int]] = (21, 24, 28)
PANEL: Final[tuple[int, int, int]] = (35, 40, 47)
PLOT_BG: Final[tuple[int, int, int]] = (17, 20, 24)
GRID: Final[tuple[int, int, int]] = (48, 55, 64)
TEXT: Final[tuple[int, int, int]] = (235, 238, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (165, 173, 184)
ACCENT: Final[tuple[int, int, int]] = (113, 204, 152)
SECONDARY: Final[tuple[int, int, int]] = (248, 183, 96)
LINK: Final[tuple[int, int, int]] = (78, 88, 99)
POINT_UNASSIGNED: Final[tuple[int, int, int]] = (210, 216, 224)
CENTROID_STROKE: Final[tuple[int, int, int]] = (13, 17, 21)

AUTO_STEP_SECONDS: Final[float] = 0.65
MIN_K: Final[int] = 2
MAX_K: Final[int] = 5
POINT_RADIUS: Final[int] = 5
CENTROID_RADIUS: Final[int] = 12

CLUSTER_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (116, 190, 240),
    (246, 177, 101),
    (168, 217, 113),
    (227, 121, 140),
    (188, 147, 238),
)


@dataclass(frozen=True, slots=True)
class Point:
    """A point in normalized plot coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class KMeansPreset:
    """Dataset preset for the intro lab."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    centers: tuple[Point, ...]
    counts: tuple[int, ...]
    spread_x: float
    spread_y: float

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


class KMeansStep(StrEnum):
    """Visible half-step of the K-Means loop."""

    ASSIGN = "assign"
    UPDATE = "update"


PRESETS: Final[tuple[KMeansPreset, ...]] = (
    KMeansPreset(
        name_en="Three blobs",
        name_pl="Trzy blobs",
        summary_en="A friendly case: compact groups and k=3 are a good match.",
        summary_pl="Przyjazny przypadek: zwarte grupy i k=3 dobrze do siebie pasują.",
        centers=(Point(-0.55, -0.42), Point(0.08, 0.44), Point(0.58, -0.18)),
        counts=(24, 24, 24),
        spread_x=0.12,
        spread_y=0.12,
    ),
    KMeansPreset(
        name_en="Two groups, k too high",
        name_pl="Dwie grupy, zbyt duże k",
        summary_en="Use k=3 or k=4 to see one natural group get split by a centroid.",
        summary_pl="Użyj k=3 albo k=4 i zobacz, jak centroid dzieli naturalną grupę.",
        centers=(Point(-0.42, -0.06), Point(0.46, 0.06)),
        counts=(34, 34),
        spread_x=0.15,
        spread_y=0.19,
    ),
    KMeansPreset(
        name_en="Stretched clusters",
        name_pl="Rozciągnięte klastry",
        summary_en="Elongated shapes show that K-Means prefers round, compact clusters.",
        summary_pl="Rozciągnięte kształty pokazują, że K-Means woli zwarte, okrągłe klastry.",
        centers=(Point(-0.55, 0.22), Point(0.02, -0.34), Point(0.58, 0.26)),
        counts=(24, 24, 24),
        spread_x=0.2,
        spread_y=0.08,
    ),
)


class KMeansIntroLabScene:
    """Introductory K-Means playground for Level 1."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic K-Means intro scene."""
        self._language = context.settings.language
        self._rng = random.Random(17)
        self._sample_seed = 17
        self._auto_elapsed = 0.0
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.k = 3
        self.step_to_run = KMeansStep.ASSIGN
        self.iteration = 0
        self.auto_run = False
        self.show_links = True
        self.points: list[Point] = []
        self.centroids: list[Point] = []
        self.assignments: list[int] = []
        self.inertia = 0.0
        self.inertia_history: list[float] = []
        self._generate_dataset()

    @property
    def preset(self) -> KMeansPreset:
        """Return the active dataset preset."""
        return PRESETS[self.preset_index]

    def handle_event(self, event: object) -> SceneCommand:
        """Handle keyboard input for the lab."""
        if isinstance(event, pygame.event.Event) and event.type == pygame.KEYDOWN:
            self._handle_keydown(event.key)
        return SceneCommand.none()

    def update(self, dt: float) -> SceneCommand:
        """Advance auto-run K-Means steps."""
        if not self.auto_run:
            return SceneCommand.none()

        self._auto_elapsed += dt
        while self._auto_elapsed >= AUTO_STEP_SECONDS:
            self._auto_elapsed -= AUTO_STEP_SECONDS
            self.step()

        return SceneCommand.none()

    def render(self, surface: object) -> None:
        """Draw the K-Means intro lab."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_plot_panel(surface, pygame.Rect(58, 128, 710, 486))
        self._draw_side_panel(surface, pygame.Rect(805, 128, 415, 486))
        self._draw_footer(surface)

    def step(self) -> None:
        """Run one visible K-Means half-step."""
        if self.step_to_run == KMeansStep.ASSIGN:
            self._assign_points()
            self.inertia = self._inertia_for_current_assignments()
            self._replace_latest_inertia()
            self.step_to_run = KMeansStep.UPDATE
            return

        self._update_centroids()
        self.iteration += 1
        self._assign_points()
        self.inertia = self._inertia_for_current_assignments()
        self.inertia_history.append(self.inertia)
        self.step_to_run = KMeansStep.ASSIGN

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
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
        elif key == pygame.K_n:
            self._sample_seed += 1
            self._generate_dataset()
        elif key == pygame.K_r:
            self.preset_index = 0
            self.k = 3
            self.auto_run = False
            self.show_links = True
            self._sample_seed = 17
            self._generate_dataset()

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "K-Means Intro Lab", (58, 38), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Step through assignment and centroid updates without labels.",
                "Przejdź przez przypisania i aktualizacje centroidów bez etykiet.",
            ),
            (58, 86),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_plot_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Points and centroids", "Punkty i centroidy"),
            (rect.x + 24, rect.y + 18),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._phase_label(),
            (rect.x + 24, rect.y + 52),
            self._font_small,
            ACCENT,
        )

        plot_rect = pygame.Rect(rect.x + 58, rect.y + 92, 590, 288)
        self._draw_kmeans_plot(surface, plot_rect)
        self._draw_inertia_strip(surface, pygame.Rect(rect.x + 58, rect.y + 404, 590, 44))
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
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            ("k", str(self.k)),
            (self._label("next step", "następny krok"), self._short_step_label()),
            (self._label("iteration", "iteracja"), str(self.iteration)),
            ("inertia", f"{self.inertia:.2f}"),
            (self._label("cluster sizes", "rozmiary klastrów"), self._cluster_sizes_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("K-Means readout", "Odczyt K-Means"),
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
                "Space: step | A: auto | -/=: k | 1-3: data | C: links | N: sample | R: reset",
                "Space: krok | A: auto | -/=: k | 1-3: dane | C: linie | N: próbka | R: reset",
            ),
            (58, 642),
            self._font_small,
            TEXT,
        )

    def _draw_kmeans_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_grid(surface, rect)

        if self.show_links:
            self._draw_centroid_links(surface, rect)

        for index, point in enumerate(self.points):
            color = self._point_color(index)
            pygame.draw.circle(surface, color, self._to_screen(rect, point), POINT_RADIUS)

        for index, centroid in enumerate(self.centroids):
            self._draw_centroid(surface, rect, centroid, CLUSTER_COLORS[index])

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_centroid_links(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for point, assignment in zip(self.points, self.assignments, strict=True):
            if assignment < 0:
                continue
            pygame.draw.line(
                surface,
                LINK,
                self._to_screen(rect, point),
                self._to_screen(rect, self.centroids[assignment]),
                1,
            )

    def _draw_centroid(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        centroid: Point,
        color: tuple[int, int, int],
    ) -> None:
        x, y = self._to_screen(rect, centroid)
        pygame.draw.circle(surface, CENTROID_STROKE, (x, y), CENTROID_RADIUS + 3)
        pygame.draw.circle(surface, color, (x, y), CENTROID_RADIUS)
        pygame.draw.line(surface, CENTROID_STROKE, (x - 7, y), (x + 7, y), 2)
        pygame.draw.line(surface, CENTROID_STROKE, (x, y - 7), (x, y + 7), 2)

    def _draw_inertia_strip(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_text(
            surface,
            "inertia",
            (rect.x + 12, rect.y + 13),
            self._font_small,
            MUTED_TEXT,
        )
        history = self.inertia_history[-12:]
        if len(history) < 2:
            return

        max_value = max(history)
        min_value = min(history)
        span = max(max_value - min_value, 0.001)
        points: list[tuple[int, int]] = []
        for index, value in enumerate(history):
            x = rect.x + 96 + round(index / max(1, len(history) - 1) * (rect.width - 120))
            y = rect.bottom - 10 - round((value - min_value) / span * (rect.height - 20))
            points.append((x, y))
        if len(points) >= 2:
            pygame.draw.lines(surface, ACCENT, False, points, 2)

    def _generate_dataset(self) -> None:
        self._rng.seed(self._sample_seed + self.preset_index * 1000)
        self.points = []
        for center, count in zip(self.preset.centers, self.preset.counts, strict=True):
            for _ in range(count):
                self.points.append(
                    Point(
                        self._clamp(center.x + self._rng.gauss(0.0, self.preset.spread_x)),
                        self._clamp(center.y + self._rng.gauss(0.0, self.preset.spread_y)),
                    ),
                )

        self._reset_centroids()

    def _reset_centroids(self) -> None:
        seed_centers = (
            Point(-0.62, -0.48),
            Point(0.0, 0.5),
            Point(0.62, -0.2),
            Point(-0.12, -0.08),
            Point(0.38, 0.42),
        )
        self.centroids = list(seed_centers[: self.k])
        self.assignments = [-1 for _ in self.points]
        self.inertia = 0.0
        self.inertia_history = []
        self.iteration = 0
        self.step_to_run = KMeansStep.ASSIGN
        self._auto_elapsed = 0.0

    def _assign_points(self) -> None:
        self.assignments = [self._nearest_centroid(point) for point in self.points]

    def _nearest_centroid(self, point: Point) -> int:
        distances = [self._squared_distance(point, centroid) for centroid in self.centroids]
        return min(range(len(distances)), key=distances.__getitem__)

    def _update_centroids(self) -> None:
        next_centroids: list[Point] = []
        for cluster_index in range(self.k):
            cluster_points = [
                point
                for point, assignment in zip(self.points, self.assignments, strict=True)
                if assignment == cluster_index
            ]
            if not cluster_points:
                next_centroids.append(self.centroids[cluster_index])
                continue
            next_centroids.append(
                Point(
                    sum(point.x for point in cluster_points) / len(cluster_points),
                    sum(point.y for point in cluster_points) / len(cluster_points),
                ),
            )
        self.centroids = next_centroids

    def _inertia_for_current_assignments(self) -> float:
        if any(assignment < 0 for assignment in self.assignments):
            return 0.0
        return sum(
            self._squared_distance(point, self.centroids[assignment])
            for point, assignment in zip(self.points, self.assignments, strict=True)
        )

    def _replace_latest_inertia(self) -> None:
        if self.inertia_history:
            self.inertia_history[-1] = self.inertia
            return
        self.inertia_history.append(self.inertia)

    def _change_k(self, delta: int) -> None:
        self.k = max(MIN_K, min(MAX_K, self.k + delta))
        self._reset_centroids()

    def _cluster_sizes_label(self) -> str:
        if any(assignment < 0 for assignment in self.assignments):
            return self._label("not assigned yet", "jeszcze bez przypisań")
        sizes = [self.assignments.count(index) for index in range(self.k)]
        return " / ".join(str(size) for size in sizes)

    def _phase_label(self) -> str:
        if self.step_to_run == KMeansStep.ASSIGN:
            return self._label(
                "Next: assign every point to the nearest centroid.",
                "Teraz: przypisz każdy punkt do najbliższego centroidu.",
            )
        return self._label(
            "Next: move each centroid to the mean of its assigned points.",
            "Teraz: przesuń każdy centroid do średniej przypisanych punktów.",
        )

    def _short_step_label(self) -> str:
        if self.step_to_run == KMeansStep.ASSIGN:
            return self._label("assign points", "przypisz punkty")
        return self._label("move centroids", "przesuń centroidy")

    def _active_takeaway(self) -> str:
        if any(assignment < 0 for assignment in self.assignments):
            return self._label(
                "Press Space to color each point by the nearest centroid.",
                "Naciśnij Space, żeby pokolorować punkty według najbliższego centroidu.",
            )
        if self.step_to_run == KMeansStep.UPDATE:
            return self._label(
                "Assignments are temporary. The next update moves centroids toward their groups.",
                "Przypisania są tymczasowe. Następny krok przesunie centroidy w stronę grup.",
            )
        if len(self.inertia_history) >= 2 and self.inertia_history[-1] <= self.inertia_history[-2]:
            return self._label(
                "Inertia dropped, so points are closer to their assigned centroids.",
                "Inertia spadła, więc punkty są bliżej przypisanych centroidów.",
            )
        return self._label(
            "Change k and compare whether the groups still look meaningful.",
            "Zmień k i sprawdź, czy grupy nadal wyglądają sensownie.",
        )

    def _point_color(self, index: int) -> tuple[int, int, int]:
        assignment = self.assignments[index]
        if assignment < 0:
            return POINT_UNASSIGNED
        return CLUSTER_COLORS[assignment]

    def _to_screen(self, rect: pygame.Rect, point: Point) -> tuple[int, int]:
        x = rect.x + round((point.x + 1.0) / 2.0 * rect.width)
        y = rect.bottom - round((point.y + 1.0) / 2.0 * rect.height)
        return (x, y)

    def _squared_distance(self, left: Point, right: Point) -> float:
        return ((left.x - right.x) ** 2) + ((left.y - right.y) ** 2)

    def _clamp(self, value: float) -> float:
        return max(-0.94, min(0.94, value))

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


def create_kmeans_intro_lab_scene(context: AppContext) -> KMeansIntroLabScene:
    """Create the unified shell K-Means Intro Lab scene."""
    return KMeansIntroLabScene(context)
