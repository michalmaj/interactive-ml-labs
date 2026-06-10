"""Native t-SNE / UMAP Exploration Lab skeleton scene for the unified shell."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Final

import pygame

from interactive_ml_labs.display import DEFAULT_RESOLUTION, Size
from interactive_ml_labs.fonts import make_ui_font
from interactive_ml_labs.scene import SceneCommand
from interactive_ml_labs.settings import AppContext

BACKGROUND: Final[tuple[int, int, int]] = (19, 22, 27)
PANEL: Final[tuple[int, int, int]] = (34, 39, 46)
PLOT_BG: Final[tuple[int, int, int]] = (15, 18, 23)
GRID: Final[tuple[int, int, int]] = (49, 56, 66)
TEXT: Final[tuple[int, int, int]] = (236, 239, 242)
MUTED_TEXT: Final[tuple[int, int, int]] = (164, 173, 184)
ACCENT: Final[tuple[int, int, int]] = (121, 203, 247)
SECONDARY: Final[tuple[int, int, int]] = (249, 185, 99)
GOOD: Final[tuple[int, int, int]] = (147, 218, 155)
WARNING: Final[tuple[int, int, int]] = (246, 132, 134)
CLASS_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (121, 203, 247),
    (249, 185, 99),
    (160, 220, 150),
)
NEIGHBOR_VALUES: Final[tuple[int, ...]] = (5, 10, 15, 30, 50)
DEFAULT_NEIGHBOR_INDEX: Final[int] = 1
ALGORITHMS: Final[tuple[str, ...]] = ("t-SNE", "UMAP")


@dataclass(frozen=True, slots=True)
class EmbeddingPreset:
    """Small deterministic high-dimensional dataset preview."""

    name_en: str
    name_pl: str
    summary_en: str
    summary_pl: str
    points: tuple[tuple[float, float, int], ...]

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


PRESETS: Final[tuple[EmbeddingPreset, ...]] = (
    EmbeddingPreset(
        name_en="Separated clusters",
        name_pl="Separated clusters",
        summary_en="Clean groups where both algorithms should look stable.",
        summary_pl="Czyste grupy, w których oba algorytmy powinny wyglądać stabilnie.",
        points=(
            (-0.72, -0.58, 0),
            (-0.64, -0.47, 0),
            (-0.82, -0.38, 0),
            (-0.55, -0.62, 0),
            (-0.70, -0.28, 0),
            (-0.46, -0.36, 0),
            (0.38, -0.52, 1),
            (0.55, -0.44, 1),
            (0.70, -0.58, 1),
            (0.46, -0.28, 1),
            (0.78, -0.24, 1),
            (0.60, -0.12, 1),
            (-0.08, 0.38, 2),
            (0.08, 0.44, 2),
            (-0.18, 0.58, 2),
            (0.20, 0.64, 2),
            (-0.02, 0.76, 2),
            (0.30, 0.48, 2),
        ),
    ),
    EmbeddingPreset(
        name_en="Bridge",
        name_pl="Bridge",
        summary_en="Two groups connected by gradual transition points.",
        summary_pl="Dwie grupy połączone punktami przejściowymi.",
        points=(
            (-0.76, -0.48, 0),
            (-0.66, -0.34, 0),
            (-0.58, -0.18, 0),
            (-0.44, -0.06, 0),
            (-0.26, 0.04, 0),
            (-0.08, 0.12, 2),
            (0.08, 0.16, 2),
            (0.26, 0.12, 2),
            (0.44, 0.04, 1),
            (0.58, -0.10, 1),
            (0.70, -0.26, 1),
            (0.80, -0.42, 1),
            (-0.30, 0.48, 2),
            (-0.08, 0.62, 2),
            (0.18, 0.58, 2),
            (0.42, 0.42, 2),
            (-0.66, 0.22, 0),
            (0.66, 0.18, 1),
        ),
    ),
    EmbeddingPreset(
        name_en="Nested groups",
        name_pl="Nested groups",
        summary_en="Local neighborhoods matter more than broad visual distance.",
        summary_pl="Lokalne sąsiedztwa są ważniejsze niż szeroka odległość na wykresie.",
        points=(
            (-0.18, -0.14, 0),
            (-0.04, -0.22, 0),
            (0.12, -0.12, 0),
            (0.18, 0.06, 0),
            (0.02, 0.18, 0),
            (-0.16, 0.10, 0),
            (-0.76, -0.08, 1),
            (-0.62, 0.34, 1),
            (-0.24, 0.70, 1),
            (0.22, 0.72, 1),
            (0.64, 0.40, 1),
            (0.78, -0.08, 1),
            (0.54, -0.54, 1),
            (0.08, -0.76, 1),
            (-0.42, -0.66, 1),
            (-0.82, -0.44, 1),
            (-0.42, 0.00, 2),
            (0.42, 0.00, 2),
        ),
    ),
)


class TSNEUMAPExplorationScene:
    """First native visual slice for the t-SNE / UMAP planning lab."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic embedding preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.preset_index = 0
        self.algorithm_index = 0
        self.neighbor_index = DEFAULT_NEIGHBOR_INDEX
        self.seed_index = 0
        self.show_links = True

    @property
    def preset(self) -> EmbeddingPreset:
        """Return the active data preset."""
        return PRESETS[self.preset_index]

    @property
    def algorithm(self) -> str:
        """Return the active embedding algorithm label."""
        return ALGORITHMS[self.algorithm_index]

    @property
    def neighbors(self) -> int:
        """Return the active neighborhood parameter."""
        return NEIGHBOR_VALUES[self.neighbor_index]

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
        """Draw the embedding preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        self._draw_embedding_panel(surface, pygame.Rect(58, 132, 540, 474))
        self._draw_comparison_panel(surface, pygame.Rect(638, 132, 286, 474))
        self._draw_side_panel(surface, pygame.Rect(960, 132, 260, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.preset_index = key - pygame.K_1
        elif key == pygame.K_m:
            self.algorithm_index = (self.algorithm_index + 1) % len(ALGORITHMS)
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self.neighbor_index = max(0, self.neighbor_index - 1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self.neighbor_index = min(len(NEIGHBOR_VALUES) - 1, self.neighbor_index + 1)
        elif key == pygame.K_s:
            self.seed_index = (self.seed_index + 1) % 4
        elif key == pygame.K_l:
            self.show_links = not self.show_links
        elif key == pygame.K_r:
            self.preset_index = 0
            self.algorithm_index = 0
            self.neighbor_index = DEFAULT_NEIGHBOR_INDEX
            self.seed_index = 0
            self.show_links = True

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "t-SNE / UMAP Exploration Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Compare deterministic toy embeddings before adding heavy algorithm dependencies.",
                "Porównaj deterministyczne toy embeddingi, zanim dodamy cięższe dependency.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_embedding_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Active embedding", "Aktywny embedding"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        plot_rect = pygame.Rect(rect.x + 44, rect.y + 78, rect.width - 88, rect.height - 136)
        self._draw_plot_frame(surface, plot_rect)
        embedded_points = self._active_embedding()
        if self.show_links:
            self._draw_neighbor_links(surface, plot_rect, embedded_points)
        self._draw_points(surface, plot_rect, embedded_points)
        self._draw_text(
            surface,
            self._label(
                "Filled groups show labels; short links show local neighborhoods.",
                "Kolory pokazują etykiety; krótkie linie pokazują lokalne sąsiedztwa.",
            ),
            (rect.x + 24, rect.bottom - 42),
            self._font_small,
            MUTED_TEXT,
        )

    def _draw_comparison_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Compare", "Porównanie"),
            (rect.x + 22, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        first_rect = pygame.Rect(rect.x + 30, rect.y + 78, rect.width - 60, 132)
        second_rect = pygame.Rect(rect.x + 30, rect.y + 248, rect.width - 60, 132)
        for algorithm_index, plot_rect in enumerate((first_rect, second_rect)):
            self._draw_plot_frame(surface, plot_rect)
            points = self._embedding_for_algorithm(algorithm_index)
            self._draw_points(surface, plot_rect, points, radius=5)
            label_color = ACCENT if algorithm_index == self.algorithm_index else MUTED_TEXT
            self._draw_text(
                surface,
                ALGORITHMS[algorithm_index],
                (plot_rect.x, plot_rect.y - 24),
                self._font_small,
                label_color,
            )
        self._draw_wrapped(
            surface,
            self._label(
                "Same labels can form different 2D stories when neighborhood assumptions change.",
                "Te same etykiety mogą tworzyć inną historię 2D, "
                "gdy zmieniają się założenia sąsiedztwa.",
            ),
            (rect.x + 22, rect.bottom - 70),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Controls", "Sterowanie"),
            (rect.x + 22, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        y = rect.y + 68
        rows = (
            (self._label("algorithm", "algorytm"), self.algorithm),
            (self._label("neighbors", "neighbors"), str(self.neighbors)),
            ("seed", str(self.seed_index)),
            (self._label("local trust", "local trust"), f"{self._local_trust_score():.0%}"),
            (self._label("global spread", "global spread"), f"{self._global_spread_score():.0%}"),
            (
                self._label("links", "linie"),
                self._label("on", "wł.") if self.show_links else self._label("off", "wył."),
            ),
        )
        for label, value in rows:
            self._draw_text(surface, f"{label}: {value}", (rect.x + 22, y), self._font_small, TEXT)
            y += 26
        y += 12
        for index, preset in enumerate(PRESETS):
            color = ACCENT if index == self.preset_index else MUTED_TEXT
            self._draw_text(
                surface,
                f"{index + 1}. {preset.name_for_language(self._language)}",
                (rect.x + 22, y),
                self._font_small,
                color,
            )
            y += 30
        self._draw_wrapped(
            surface,
            self.preset.summary_for_language(self._language),
            (rect.x + 22, rect.bottom - 76),
            rect.width - 44,
            self._font_small,
            MUTED_TEXT,
            line_height=17,
        )

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: compare neighborhood structure without over-reading every 2D distance.",
                "Cel: porównuj strukturę sąsiedztwa bez nadinterpretowania każdej odległości 2D.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _active_embedding(self) -> tuple[tuple[float, float, int], ...]:
        return self._embedding_for_algorithm(self.algorithm_index)

    def _embedding_for_algorithm(
        self, algorithm_index: int
    ) -> tuple[tuple[float, float, int], ...]:
        raw_points = self.preset.points
        shaped: list[tuple[float, float, int]] = []
        for point_index, (x, y, label) in enumerate(raw_points):
            seed_shift = (self.seed_index - 1.5) * 0.035
            neighbor_shift = (self.neighbors - 15) / 70
            if algorithm_index == 0:
                angle = (point_index % 6) * pi / 18 + self.seed_index * 0.08
                new_x = x * (1.0 + neighbor_shift * 0.18) + sin(angle) * 0.045 + seed_shift
                new_y = y * (0.86 - neighbor_shift * 0.08) + cos(angle) * 0.045 - seed_shift
            else:
                radial = 0.90 + neighbor_shift * 0.14
                new_x = x * radial + y * 0.12 + seed_shift * 0.5
                new_y = y * (1.0 + neighbor_shift * 0.10) - x * 0.10 - seed_shift * 0.5
            shaped.append((max(-0.95, min(0.95, new_x)), max(-0.95, min(0.95, new_y)), label))
        return tuple(shaped)

    def _nearest_neighbor_pairs(
        self, points: tuple[tuple[float, float, int], ...]
    ) -> tuple[tuple[int, int], ...]:
        pairs: list[tuple[int, int]] = []
        for index, point in enumerate(points):
            nearest_index = min(
                (other_index for other_index in range(len(points)) if other_index != index),
                key=lambda other_index: self._squared_distance(point, points[other_index]),
            )
            pair = tuple(sorted((index, nearest_index)))
            if pair not in pairs:
                pairs.append(pair)
        return tuple(pairs)

    def _local_trust_score(self) -> float:
        points = self._active_embedding()
        pairs = self._nearest_neighbor_pairs(points)
        same_label_pairs = sum(1 for left, right in pairs if points[left][2] == points[right][2])
        return same_label_pairs / len(pairs)

    def _global_spread_score(self) -> float:
        points = self._active_embedding()
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        spread = ((max(xs) - min(xs)) + (max(ys) - min(ys))) / 4
        return max(0.0, min(1.0, spread))

    def _draw_neighbor_links(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        points: tuple[tuple[float, float, int], ...],
    ) -> None:
        for left, right in self._nearest_neighbor_pairs(points):
            left_pos = self._plot_position(points[left], rect)
            right_pos = self._plot_position(points[right], rect)
            pygame.draw.line(surface, GRID, left_pos, right_pos, 1)

    def _draw_points(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        points: tuple[tuple[float, float, int], ...],
        *,
        radius: int = 8,
    ) -> None:
        for point in points:
            color = CLASS_COLORS[point[2] % len(CLASS_COLORS)]
            pygame.draw.circle(surface, color, self._plot_position(point, rect), radius)
            pygame.draw.circle(surface, PLOT_BG, self._plot_position(point, rect), radius, width=1)

    def _plot_position(self, point: tuple[float, float, int], rect: pygame.Rect) -> tuple[int, int]:
        x, y, _label = point
        return (
            rect.centerx + round(x * rect.width * 0.46),
            rect.centery - round(y * rect.height * 0.46),
        )

    def _squared_distance(
        self, left: tuple[float, float, int], right: tuple[float, float, int]
    ) -> float:
        return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

    def _draw_plot_frame(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        pygame.draw.line(surface, GRID, (rect.left, rect.centery), (rect.right, rect.centery), 1)
        pygame.draw.line(surface, GRID, (rect.centerx, rect.top), (rect.centerx, rect.bottom), 1)

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


def create_tsne_umap_exploration_scene(context: AppContext) -> TSNEUMAPExplorationScene:
    """Create the unified shell t-SNE / UMAP Exploration Lab scene."""
    return TSNEUMAPExplorationScene(context)
