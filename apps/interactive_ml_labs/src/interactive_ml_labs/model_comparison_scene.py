"""Native Model Comparison Lab skeleton scene for the unified shell."""

from __future__ import annotations

import math
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
CLASS_A: Final[tuple[int, int, int]] = (116, 184, 255)
CLASS_B: Final[tuple[int, int, int]] = (247, 179, 101)
LOGISTIC: Final[tuple[int, int, int]] = (118, 205, 247)
KNN: Final[tuple[int, int, int]] = (151, 219, 156)
TREE: Final[tuple[int, int, int]] = (244, 131, 133)
BOUNDARY_MUTED: Final[tuple[int, int, int]] = (82, 91, 103)


@dataclass(frozen=True, slots=True)
class ComparisonModel:
    """Static model preview metadata."""

    name: str
    shortcut: str
    color: tuple[int, int, int]
    assumption_en: str
    assumption_pl: str
    boundary_en: str
    boundary_pl: str
    risk_en: str
    risk_pl: str

    def assumption_for_language(self, language: str) -> str:
        """Return localized model assumption."""
        if language == "pl":
            return self.assumption_pl

        return self.assumption_en

    def boundary_for_language(self, language: str) -> str:
        """Return localized boundary description."""
        if language == "pl":
            return self.boundary_pl

        return self.boundary_en

    def risk_for_language(self, language: str) -> str:
        """Return localized risk description."""
        if language == "pl":
            return self.risk_pl

        return self.risk_en


MODELS: Final[tuple[ComparisonModel, ...]] = (
    ComparisonModel(
        name="Logistic Regression",
        shortcut="1",
        color=LOGISTIC,
        assumption_en="linear score, one smooth global split",
        assumption_pl="liniowy score i jeden globalny podział",
        boundary_en="straight decision boundary",
        boundary_pl="prosta decision boundary",
        risk_en="can underfit curved patterns",
        risk_pl="może nie złapać zakrzywionego wzorca",
    ),
    ComparisonModel(
        name="k-NN",
        shortcut="2",
        color=KNN,
        assumption_en="nearby points should vote similarly",
        assumption_pl="bliskie punkty powinny głosować podobnie",
        boundary_en="local, flexible boundary",
        boundary_pl="lokalna, elastyczna decision boundary",
        risk_en="can follow noise too closely",
        risk_pl="może za mocno podążać za noise",
    ),
    ComparisonModel(
        name="Decision Tree",
        shortcut="3",
        color=TREE,
        assumption_en="axis-aligned rules explain the data",
        assumption_pl="reguły wzdłuż osi wyjaśniają dane",
        boundary_en="blocky step boundary",
        boundary_pl="blokowa, schodkowa decision boundary",
        risk_en="can memorize small regions",
        risk_pl="może zapamiętywać małe regiony",
    ),
)

POINTS: Final[tuple[tuple[float, float, int], ...]] = (
    (-0.82, -0.36, 0),
    (-0.72, -0.12, 0),
    (-0.62, -0.52, 0),
    (-0.52, 0.10, 0),
    (-0.42, -0.26, 0),
    (-0.32, 0.32, 0),
    (-0.24, -0.04, 0),
    (-0.12, 0.42, 0),
    (0.02, 0.18, 0),
    (0.14, 0.54, 0),
    (0.28, 0.24, 0),
    (0.42, 0.64, 0),
    (-0.40, 0.70, 1),
    (-0.20, 0.78, 1),
    (0.02, 0.78, 1),
    (0.22, 0.70, 1),
    (0.42, 0.46, 1),
    (0.58, 0.22, 1),
    (0.68, -0.02, 1),
    (0.72, -0.32, 1),
    (0.58, -0.58, 1),
    (0.34, -0.64, 1),
    (0.10, -0.54, 1),
    (-0.04, -0.30, 1),
)


class ModelComparisonLabScene:
    """First interactive slice for comparing classifier boundary shapes."""

    fixed_scene_size: Size = DEFAULT_RESOLUTION

    def __init__(self, context: AppContext) -> None:
        """Create a deterministic model comparison preview scene."""
        self._language = context.settings.language
        self._font_title = make_ui_font(34, bold=True)
        self._font_heading = make_ui_font(23, bold=True)
        self._font_body = make_ui_font(18)
        self._font_small = make_ui_font(15)
        self.selected_model_index = 0
        self.show_all_boundaries = True

    @property
    def selected_model(self) -> ComparisonModel:
        """Return the active model preview."""
        return MODELS[self.selected_model_index]

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
        """Draw the Model Comparison preview."""
        if not isinstance(surface, pygame.Surface):
            return

        surface.fill(BACKGROUND)
        self._draw_header(surface)
        plot_rect = pygame.Rect(58, 132, 720, 474)
        self._draw_plot(surface, plot_rect)
        self._draw_side_panel(surface, pygame.Rect(818, 132, 390, 474))
        self._draw_footer(surface)

    def _handle_keydown(self, key: int) -> None:
        if key in {pygame.K_1, pygame.K_2, pygame.K_3}:
            self.selected_model_index = key - pygame.K_1
        elif key == pygame.K_a:
            self.show_all_boundaries = not self.show_all_boundaries
        elif key == pygame.K_r:
            self.selected_model_index = 0
            self.show_all_boundaries = True

    def _draw_header(self, surface: pygame.Surface) -> None:
        self._draw_text(surface, "Model Comparison Lab", (58, 40), self._font_title, TEXT)
        self._draw_text(
            surface,
            self._label(
                "Compare how model assumptions shape decision boundaries on the same dataset.",
                "Porównaj, jak założenia modeli zmieniają decision boundary na tych samych danych.",
            ),
            (58, 88),
            self._font_body,
            MUTED_TEXT,
        )

    def _draw_plot(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Same dataset, different boundaries", "Te same dane, różne granice"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        plot_rect = pygame.Rect(rect.x + 36, rect.y + 76, rect.width - 72, rect.height - 124)
        self._draw_grid(surface, plot_rect)
        self._draw_boundaries(surface, plot_rect)
        self._draw_points(surface, plot_rect)
        self._draw_plot_legend(surface, plot_rect)

    def _draw_side_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_panel(surface, rect)
        self._draw_text(
            surface,
            self._label("Model preview", "Podgląd modelu"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        y = rect.y + 72
        for index, model in enumerate(MODELS):
            active = index == self.selected_model_index
            item_rect = pygame.Rect(rect.x + 22, y, rect.width - 44, 86)
            if active:
                pygame.draw.rect(surface, (45, 53, 62), item_rect, border_radius=8)
                pygame.draw.rect(surface, model.color, item_rect, width=2, border_radius=8)
            else:
                pygame.draw.rect(surface, (28, 32, 37), item_rect, border_radius=8)
            self._draw_text(
                surface,
                f"{model.shortcut}. {model.name}",
                (item_rect.x + 16, item_rect.y + 12),
                self._font_body,
                model.color if active else TEXT,
            )
            self._draw_wrapped(
                surface,
                model.assumption_for_language(self._language),
                (item_rect.x + 16, item_rect.y + 42),
                item_rect.width - 32,
                self._font_small,
                MUTED_TEXT,
                line_height=18,
            )
            y += 100
        self._draw_selected_model_details(
            surface, pygame.Rect(rect.x + 24, y + 8, rect.width - 48, 122)
        )

    def _draw_selected_model_details(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        model = self.selected_model
        self._draw_text(
            surface,
            self._label("Read the boundary", "Jak czytać granicę"),
            (rect.x, rect.y),
            self._font_body,
            TEXT,
        )
        details = (
            f"{self._label('Shape', 'Kształt')}: {model.boundary_for_language(self._language)}",
            f"{self._label('Watch for', 'Uważaj na')}: {model.risk_for_language(self._language)}",
            self._label(
                "A toggles all boundaries. R resets the preview.",
                "A przełącza wszystkie granice. R resetuje podgląd.",
            ),
        )
        y = rect.y + 34
        for line in details:
            self._draw_wrapped(
                surface,
                line,
                (rect.x, y),
                rect.width,
                self._font_small,
                MUTED_TEXT,
                line_height=18,
            )
            y += 38

    def _draw_footer(self, surface: pygame.Surface) -> None:
        self._draw_text(
            surface,
            self._label(
                "Goal: separate model quality from model shape before comparing metrics.",
                "Cel: najpierw rozróżnij jakość modelu i kształt granicy, dopiero potem metryki.",
            ),
            (58, 635),
            self._font_body,
            TEXT,
        )

    def _draw_boundaries(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        active_index = self.selected_model_index
        for index, draw_boundary in enumerate(
            (
                self._draw_logistic_boundary,
                self._draw_knn_boundary,
                self._draw_tree_boundary,
            )
        ):
            if not self.show_all_boundaries and index != active_index:
                continue
            model = MODELS[index]
            color = model.color if index == active_index else BOUNDARY_MUTED
            width = 5 if index == active_index else 2
            draw_boundary(surface, rect, color, width)

    def _draw_logistic_boundary(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        start = self._to_screen((-0.78, 0.92), rect)
        end = self._to_screen((0.82, -0.78), rect)
        pygame.draw.line(surface, color, start, end, width)

    def _draw_knn_boundary(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        points = tuple(
            self._to_screen(
                (
                    -0.78 + index * 0.18,
                    0.34 * math.sin(index * 0.9) - 0.16 + index * 0.018,
                ),
                rect,
            )
            for index in range(10)
        )
        pygame.draw.lines(surface, color, False, points, width)

    def _draw_tree_boundary(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        points = (
            self._to_screen((-0.72, 0.68), rect),
            self._to_screen((-0.36, 0.68), rect),
            self._to_screen((-0.36, 0.18), rect),
            self._to_screen((0.12, 0.18), rect),
            self._to_screen((0.12, -0.34), rect),
            self._to_screen((0.56, -0.34), rect),
            self._to_screen((0.56, -0.76), rect),
        )
        pygame.draw.lines(surface, color, False, points, width)

    def _draw_points(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for x, y, class_id in POINTS:
            color = CLASS_A if class_id == 0 else CLASS_B
            pygame.draw.circle(surface, color, self._to_screen((x, y), rect), 7)
            pygame.draw.circle(surface, PLOT_BG, self._to_screen((x, y), rect), 7, width=1)

    def _draw_plot_legend(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        y = rect.bottom + 18
        labels = (
            (CLASS_A, self._label("class A", "klasa A")),
            (CLASS_B, self._label("class B", "klasa B")),
            (self.selected_model.color, self.selected_model.name),
        )
        x = rect.x + 10
        for color, label in labels:
            pygame.draw.circle(surface, color, (x, y + 8), 6)
            self._draw_text(surface, label, (x + 16, y), self._font_small, MUTED_TEXT)
            x += self._font_small.size(label)[0] + 54

    def _draw_grid(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        pygame.draw.rect(surface, PLOT_BG, rect, border_radius=6)
        pygame.draw.rect(surface, GRID, rect, width=1, border_radius=6)
        for step in range(1, 4):
            x = rect.left + round(step * rect.width / 4)
            y = rect.top + round(step * rect.height / 4)
            pygame.draw.line(surface, GRID, (x, rect.top), (x, rect.bottom), 1)
            pygame.draw.line(surface, GRID, (rect.left, y), (rect.right, y), 1)

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

    def _to_screen(self, point: tuple[float, float], rect: pygame.Rect) -> tuple[int, int]:
        x, y = point
        return (
            rect.left + round((x + 1.0) * 0.5 * rect.width),
            rect.top + round((1.0 - (y + 1.0) * 0.5) * rect.height),
        )

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl

        return en


def create_model_comparison_lab_scene(context: AppContext) -> ModelComparisonLabScene:
    """Create the unified shell Model Comparison Lab scene."""
    return ModelComparisonLabScene(context)
