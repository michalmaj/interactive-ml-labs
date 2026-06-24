"""Native Gaussian Mixture Intro Lab scene for the unified shell."""

from __future__ import annotations

import math
import random
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

MIN_COMPONENTS: Final[int] = 2
MAX_COMPONENTS: Final[int] = 4
QUERY_STEP: Final[float] = 0.08
MIN_CONFIDENCE: Final[float] = 0.55
GAUSSIAN_MIXTURE_LESSON_ID: Final[str] = "distance_soft_clusters"
COMPARE_SOFT_ASSIGNMENTS_TASK_ID: Final[str] = "compare_soft_assignments"
ADJUST_COMPONENT_COUNT_TASK_ID: Final[str] = "adjust_component_count"

COMPONENT_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (116, 190, 240),
    (246, 177, 101),
    (169, 217, 113),
    (188, 147, 238),
)


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in normalized plot coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class GaussianComponent:
    """One diagonal-covariance Gaussian component."""

    mean: Point
    sigma_x: float
    sigma_y: float
    weight: float


@dataclass(frozen=True, slots=True)
class MixturePreset:
    """Static Gaussian mixture scenario."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    components: tuple[GaussianComponent, ...]
    query_start: Point

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


PRESETS: Final[tuple[MixturePreset, ...]] = (
    MixturePreset(
        name_en="Overlapping customers",
        name_pl="Nakładający się klienci",
        summary_en=(
            "Two groups overlap, so the query point can carry meaningful responsibility "
            "for both components."
        ),
        summary_pl=(
            "Dwie grupy nachodzą na siebie, więc query point może mieć sensowną "
            "responsibility dla obu komponentów."
        ),
        components=(
            GaussianComponent(Point(-0.38, -0.04), 0.28, 0.2, 0.56),
            GaussianComponent(Point(0.36, 0.12), 0.26, 0.22, 0.44),
            GaussianComponent(Point(0.7, -0.55), 0.16, 0.14, 0.0),
            GaussianComponent(Point(-0.72, 0.52), 0.17, 0.16, 0.0),
        ),
        query_start=Point(0.02, 0.03),
    ),
    MixturePreset(
        name_en="Rare segment",
        name_pl="Rzadki segment",
        summary_en=(
            "A small component can explain a local pocket even when its mixture weight is low."
        ),
        summary_pl=(
            "Mały komponent może tłumaczyć lokalną kieszeń danych, nawet gdy jego "
            "mixture weight jest niski."
        ),
        components=(
            GaussianComponent(Point(-0.52, -0.32), 0.23, 0.2, 0.68),
            GaussianComponent(Point(0.36, 0.28), 0.2, 0.18, 0.22),
            GaussianComponent(Point(0.66, -0.46), 0.13, 0.12, 0.1),
            GaussianComponent(Point(-0.76, 0.55), 0.16, 0.16, 0.0),
        ),
        query_start=Point(0.57, -0.39),
    ),
    MixturePreset(
        name_en="Different shapes",
        name_pl="Różne kształty",
        summary_en=(
            "Diagonal covariance changes component shape, so distance to the mean is "
            "not the whole story."
        ),
        summary_pl=(
            "Diagonal covariance zmienia kształt komponentu, więc dystans do mean "
            "nie jest całą historią."
        ),
        components=(
            GaussianComponent(Point(-0.48, 0.08), 0.38, 0.12, 0.36),
            GaussianComponent(Point(0.12, -0.28), 0.16, 0.34, 0.34),
            GaussianComponent(Point(0.58, 0.36), 0.2, 0.18, 0.3),
            GaussianComponent(Point(-0.74, -0.58), 0.14, 0.14, 0.0),
        ),
        query_start=Point(-0.1, 0.0),
    ),
)


class GaussianMixtureIntroLabScene:
    """Introductory Gaussian Mixture Model playground for Level 2."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic Gaussian Mixture intro scene."""
        self._context = context
        self._language = context.settings.language
        self._rng = random.Random(23)
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.component_count = 2
        self.show_density = True
        self.hard_assignment = False
        self.query = PRESETS[0].query_start
        self.points = self._sample_points()

    @property
    def preset(self) -> MixturePreset:
        """Return the active mixture preset."""
        return PRESETS[self.preset_index]

    @property
    def active_components(self) -> tuple[GaussianComponent, ...]:
        """Return active components with renormalized weights."""
        components = self.preset.components[: self.component_count]
        total_weight = sum(component.weight for component in components)
        if total_weight <= 0:
            return components
        return tuple(
            GaussianComponent(
                component.mean,
                component.sigma_x,
                component.sigma_y,
                component.weight / total_weight,
            )
            for component in components
        )

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
        """Draw the Gaussian Mixture intro lab."""
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
            self.component_count = 2
            self.query = self.preset.query_start
            self.points = self._sample_points()
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            previous_count = self.component_count
            self.component_count = max(MIN_COMPONENTS, self.component_count - 1)
            if self.component_count != previous_count:
                self._record_component_count_progress()
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            previous_count = self.component_count
            self.component_count = min(MAX_COMPONENTS, self.component_count + 1)
            if self.component_count != previous_count:
                self._record_component_count_progress()
        elif key == pygame.K_h:
            self.hard_assignment = not self.hard_assignment
        elif key == pygame.K_d:
            self.show_density = not self.show_density
        elif key == pygame.K_LEFT:
            self._move_query(-QUERY_STEP, 0.0)
            self._record_soft_assignment_progress()
        elif key == pygame.K_RIGHT:
            self._move_query(QUERY_STEP, 0.0)
            self._record_soft_assignment_progress()
        elif key == pygame.K_UP:
            self._move_query(0.0, QUERY_STEP)
            self._record_soft_assignment_progress()
        elif key == pygame.K_DOWN:
            self._move_query(0.0, -QUERY_STEP)
            self._record_soft_assignment_progress()
        elif key == pygame.K_r:
            self.preset_index = 0
            self.component_count = 2
            self.show_density = True
            self.hard_assignment = False
            self.query = self.preset.query_start
            self.points = self._sample_points()

    def _record_soft_assignment_progress(self) -> None:
        """Complete the soft-assignment comparison task for the guided lesson."""
        if self._context.selected_lesson_id != GAUSSIAN_MIXTURE_LESSON_ID:
            return

        self._context.progress.complete_task(
            GAUSSIAN_MIXTURE_LESSON_ID,
            COMPARE_SOFT_ASSIGNMENTS_TASK_ID,
        )
        self._mark_lesson_completed_if_ready()

    def _record_component_count_progress(self) -> None:
        """Complete the component-count adjustment task for the guided lesson."""
        if self._context.selected_lesson_id != GAUSSIAN_MIXTURE_LESSON_ID:
            return

        self._context.progress.complete_task(
            GAUSSIAN_MIXTURE_LESSON_ID,
            ADJUST_COMPONENT_COUNT_TASK_ID,
        )
        self._mark_lesson_completed_if_ready()

    def _mark_lesson_completed_if_ready(self) -> None:
        """Complete the lesson once both tasks are done."""
        progress = self._context.progress.lessons.get(GAUSSIAN_MIXTURE_LESSON_ID)
        if progress is None:
            return

        required_tasks = {COMPARE_SOFT_ASSIGNMENTS_TASK_ID, ADJUST_COMPONENT_COUNT_TASK_ID}
        if required_tasks.issubset(progress.completed_task_ids):
            self._context.progress.mark_completed(GAUSSIAN_MIXTURE_LESSON_ID)

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Gaussian Mixture Intro Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Move a query point and watch soft component responsibilities change.",
                "Przesuwaj query point i obserwuj soft responsibilities komponentów.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_plot_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Soft clusters", "Miękkie klastry"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self._mode_label(),
            (rect.x + 24, rect.y + 54),
            self._font_small,
            GOOD if not self.hard_assignment else WARNING,
        )
        plot_rect = pygame.Rect(rect.x + 64, rect.y + 102, 570, 286)
        self._draw_mixture_plot(surface, plot_rect)
        self._draw_responsibility_bars(surface, pygame.Rect(rect.x + 74, rect.y + 414, 550, 36))
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
        responsibilities = self.responsibilities_for_query()
        winner = self._winner_index(responsibilities)
        rows = (
            (self._label("dataset", "dataset"), self.preset.name_for_language(self._language)),
            (self._label("components", "komponenty"), str(self.component_count)),
            (self._label("top component", "główny komponent"), f"{winner + 1}"),
            (
                self._label("top responsibility", "top responsibility"),
                f"{responsibilities[winner]:.0%}",
            ),
            (self._label("mixture weight", "mixture weight"), self._weights_label()),
            (self._label("mode", "tryb"), self._mode_label()),
            (self._label("diagnosis", "diagnoza"), self._diagnosis_label()),
        )
        options = tuple(
            (f"{index + 1}. {preset.name_for_language(self._language)}", index == self.preset_index)
            for index, preset in enumerate(PRESETS)
        )
        draw_readout_panel(
            surface,
            rect,
            title=self._label("GMM readout", "Odczyt GMM"),
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
                "Arrows: query | -/=: components | H: hard/soft | D: density | 1-3: data",
                "Strzałki: query | -/=: komponenty | H: hard/soft | D: density | 1-3: dane",
            ),
            (58, 642),
            self._font_small,
            TEXT,
        )

    def _draw_mixture_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        self._draw_grid(surface, rect)

        if self.show_density:
            self._draw_density_ellipses(surface, rect)

        for point, assignment in self.points:
            color = COMPONENT_COLORS[assignment]
            pygame.draw.circle(surface, color, self._to_screen(rect, point), 4)

        for index, component in enumerate(self.active_components):
            self._draw_component_center(surface, rect, component, COMPONENT_COLORS[index])

        query_position = self._to_screen(rect, self.query)
        pygame.draw.circle(surface, TEXT, query_position, 9)
        pygame.draw.circle(surface, BACKGROUND, query_position, 5)
        pygame.draw.circle(surface, TEXT, query_position, 2)

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for step in range(1, 4):
            x = rect.x + round(rect.width * step / 4)
            y = rect.y + round(rect.height * step / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

    def _draw_density_ellipses(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for index, component in enumerate(self.active_components):
            center = self._to_screen(rect, component.mean)
            width = round(component.sigma_x * rect.width * 2.8)
            height = round(component.sigma_y * rect.height * 2.8)
            ellipse = pygame.Rect(0, 0, width, height)
            ellipse.center = center
            pygame.draw.ellipse(surface, COMPONENT_COLORS[index], ellipse, width=2)

    def _draw_component_center(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        component: GaussianComponent,
        color: tuple[int, int, int],
    ) -> None:
        x, y = self._to_screen(rect, component.mean)
        pygame.draw.circle(surface, BACKGROUND, (x, y), 11)
        pygame.draw.circle(surface, color, (x, y), 8)
        pygame.draw.circle(surface, TEXT, (x, y), 2)

    def _draw_responsibility_bars(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        responsibilities = self.responsibilities_for_query()
        if self.hard_assignment:
            winner = self._winner_index(responsibilities)
            responsibilities = tuple(
                1.0 if index == winner else 0.0 for index in range(len(responsibilities))
            )

        x = rect.x
        for index, value in enumerate(responsibilities):
            width = round(rect.width * value)
            if width > 0:
                pygame.draw.rect(
                    surface,
                    COMPONENT_COLORS[index],
                    pygame.Rect(x, rect.y, width, rect.height),
                    border_radius=4 if value >= 0.98 else 0,
                )
            x += width
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=4)
        self._draw_text(
            surface,
            self._label("responsibilities", "responsibilities"),
            (rect.x, rect.y - 24),
            self._font_small,
            MUTED_TEXT,
        )

    def _sample_points(self) -> list[tuple[Point, int]]:
        self._rng.seed(23 + self.preset_index * 1000)
        points: list[tuple[Point, int]] = []
        for index, component in enumerate(self.active_components):
            count = max(8, round(component.weight * 90))
            for _ in range(count):
                points.append(
                    (
                        Point(
                            self._clamp(component.mean.x + self._rng.gauss(0.0, component.sigma_x)),
                            self._clamp(component.mean.y + self._rng.gauss(0.0, component.sigma_y)),
                        ),
                        index,
                    )
                )
        return points

    def responsibilities_for_query(self) -> tuple[float, ...]:
        """Return posterior component responsibilities for the query point."""
        scores = tuple(
            component.weight * self._gaussian_density(self.query, component)
            for component in self.active_components
        )
        total = sum(scores)
        if total <= 0:
            return tuple(1 / len(scores) for _ in scores)
        return tuple(score / total for score in scores)

    def _gaussian_density(self, point: Point, component: GaussianComponent) -> float:
        normalized_x = ((point.x - component.mean.x) / component.sigma_x) ** 2
        normalized_y = ((point.y - component.mean.y) / component.sigma_y) ** 2
        normalization = 2 * math.pi * component.sigma_x * component.sigma_y
        return math.exp(-0.5 * (normalized_x + normalized_y)) / normalization

    def _winner_index(self, responsibilities: tuple[float, ...]) -> int:
        return max(range(len(responsibilities)), key=responsibilities.__getitem__)

    def _move_query(self, dx: float, dy: float) -> None:
        self.query = Point(self._clamp(self.query.x + dx), self._clamp(self.query.y + dy))

    def _weights_label(self) -> str:
        return " / ".join(f"{component.weight:.0%}" for component in self.active_components)

    def _mode_label(self) -> str:
        if self.hard_assignment:
            return self._label("hard assignment", "hard assignment")
        return self._label("soft responsibilities", "soft responsibilities")

    def _diagnosis_key(self) -> str:
        top = max(self.responsibilities_for_query())
        if top < MIN_CONFIDENCE:
            return "ambiguous"
        if top > 0.85:
            return "confident"
        return "mixed"

    def _diagnosis_label(self) -> str:
        key = self._diagnosis_key()
        if key == "ambiguous":
            return self._label("ambiguous point", "punkt niejednoznaczny")
        if key == "confident":
            return self._label("clear component", "czytelny komponent")
        return self._label("mixed evidence", "mieszany sygnał")

    def _active_takeaway(self) -> str:
        key = self._diagnosis_key()
        if self.hard_assignment:
            return self._label(
                "Hard assignment hides uncertainty by forcing one component to win.",
                "Hard assignment ukrywa uncertainty, bo wymusza wygraną jednego komponentu.",
            )
        if key == "ambiguous":
            return self._label(
                "The query sits between components, so soft responsibilities are useful.",
                "Query point leży między komponentami, więc soft responsibilities pomagają.",
            )
        if key == "confident":
            return self._label(
                "One component explains the query much better than the others.",
                "Jeden komponent tłumaczy query point dużo lepiej niż pozostałe.",
            )
        return self._label(
            "Several components still have plausible responsibility for this point.",
            "Kilka komponentów nadal ma wiarygodną responsibility dla tego punktu.",
        )

    def _to_screen(self, rect: pygame.Rect, point: Point) -> tuple[int, int]:
        x = rect.x + round((point.x + 1.0) / 2.0 * rect.width)
        y = rect.bottom - round((point.y + 1.0) / 2.0 * rect.height)
        return (x, y)

    def _clamp(self, value: float) -> float:
        return max(-0.94, min(0.94, value))

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


def create_gaussian_mixture_intro_lab_scene(
    context: AppContext,
) -> GaussianMixtureIntroLabScene:
    """Create the unified shell Gaussian Mixture Intro Lab scene."""
    return GaussianMixtureIntroLabScene(context)
