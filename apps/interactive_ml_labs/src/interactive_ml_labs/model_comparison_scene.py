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
TEST_POINT_OUTLINE: Final[tuple[int, int, int]] = (236, 239, 242)
ERROR_HIGHLIGHT: Final[tuple[int, int, int]] = (255, 99, 99)
DEFAULT_COMPLEXITY_LEVEL: Final[int] = 1
MIN_COMPLEXITY_LEVEL: Final[int] = 0
MAX_COMPLEXITY_LEVEL: Final[int] = 2
KNN_K_VALUES: Final[tuple[int, int, int]] = (9, 5, 1)
LOGISTIC_THRESHOLDS: Final[tuple[float, float, float]] = (0.35, 0.50, 0.65)
TREE_DEPTHS: Final[tuple[int, int, int]] = (1, 2, 3)


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


@dataclass(frozen=True, slots=True)
class ComparisonDataset:
    """Static dataset preset for the comparison preview."""

    name_en: str
    name_pl: str
    key: str
    points: tuple[tuple[float, float, int], ...]

    def name_for_language(self, language: str) -> str:
        """Return a localized dataset name."""
        if language == "pl":
            return self.name_pl

        return self.name_en


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

CURVED_POINTS: Final[tuple[tuple[float, float, int], ...]] = (
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

LINEAR_POINTS: Final[tuple[tuple[float, float, int], ...]] = (
    (-0.86, -0.70, 0),
    (-0.72, -0.48, 0),
    (-0.62, -0.72, 0),
    (-0.48, -0.30, 0),
    (-0.32, -0.54, 0),
    (-0.18, -0.18, 0),
    (0.02, -0.34, 0),
    (0.18, -0.08, 0),
    (0.36, -0.22, 0),
    (0.52, 0.04, 0),
    (0.72, -0.02, 0),
    (0.84, 0.18, 0),
    (-0.84, -0.10, 1),
    (-0.70, 0.18, 1),
    (-0.54, 0.02, 1),
    (-0.36, 0.34, 1),
    (-0.18, 0.16, 1),
    (0.00, 0.46, 1),
    (0.18, 0.30, 1),
    (0.34, 0.60, 1),
    (0.52, 0.44, 1),
    (0.66, 0.72, 1),
    (0.80, 0.54, 1),
    (0.90, 0.82, 1),
)

OVERLAP_POINTS: Final[tuple[tuple[float, float, int], ...]] = (
    (-0.82, -0.28, 0),
    (-0.68, 0.02, 0),
    (-0.52, -0.44, 0),
    (-0.42, 0.24, 0),
    (-0.24, -0.10, 0),
    (-0.08, 0.28, 0),
    (0.08, -0.32, 0),
    (0.22, 0.10, 0),
    (0.38, -0.18, 0),
    (0.54, 0.18, 0),
    (0.70, -0.06, 0),
    (0.82, 0.36, 0),
    (-0.76, 0.34, 1),
    (-0.58, 0.58, 1),
    (-0.40, 0.06, 1),
    (-0.22, 0.50, 1),
    (-0.02, 0.00, 1),
    (0.12, 0.52, 1),
    (0.28, -0.02, 1),
    (0.42, 0.44, 1),
    (0.58, -0.34, 1),
    (0.66, 0.62, 1),
    (0.78, -0.46, 1),
    (0.88, 0.08, 1),
)

DATASETS: Final[tuple[ComparisonDataset, ...]] = (
    ComparisonDataset("Curved boundary", "Zakrzywiona granica", "curved", CURVED_POINTS),
    ComparisonDataset("Linear signal", "Liniowy sygnał", "linear", LINEAR_POINTS),
    ComparisonDataset("Noisy overlap", "Szum i overlap", "overlap", OVERLAP_POINTS),
)
POINTS: Final[tuple[tuple[float, float, int], ...]] = CURVED_POINTS


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
        self.selected_dataset_index = 0
        self.complexity_levels = [DEFAULT_COMPLEXITY_LEVEL for _model in MODELS]
        self.show_all_boundaries = True
        self.show_error_highlights = True

    @property
    def selected_model(self) -> ComparisonModel:
        """Return the active model preview."""
        return MODELS[self.selected_model_index]

    @property
    def selected_dataset(self) -> ComparisonDataset:
        """Return the active dataset preset."""
        return DATASETS[self.selected_dataset_index]

    @property
    def points(self) -> tuple[tuple[float, float, int], ...]:
        """Return points for the active dataset."""
        return self.selected_dataset.points

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
        elif key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
            self._change_selected_complexity(-1)
        elif key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
            self._change_selected_complexity(1)
        elif key == pygame.K_d:
            self.selected_dataset_index = (self.selected_dataset_index + 1) % len(DATASETS)
        elif key == pygame.K_a:
            self.show_all_boundaries = not self.show_all_boundaries
        elif key == pygame.K_e:
            self.show_error_highlights = not self.show_error_highlights
        elif key == pygame.K_r:
            self.selected_model_index = 0
            self.selected_dataset_index = 0
            self.complexity_levels = [DEFAULT_COMPLEXITY_LEVEL for _model in MODELS]
            self.show_all_boundaries = True
            self.show_error_highlights = True

    def _change_selected_complexity(self, delta: int) -> None:
        current_level = self.complexity_levels[self.selected_model_index]
        self.complexity_levels[self.selected_model_index] = max(
            MIN_COMPLEXITY_LEVEL,
            min(MAX_COMPLEXITY_LEVEL, current_level + delta),
        )

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
        self._draw_text(
            surface,
            f"{self._label('dataset', 'dataset')}: "
            f"{self.selected_dataset.name_for_language(self._language)}",
            (rect.right - 260, rect.y + 25),
            self._font_small,
            MUTED_TEXT,
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
            self._label("Models and score", "Modele i wynik"),
            (rect.x + 24, rect.y + 20),
            self._font_heading,
            TEXT,
        )
        self._draw_text(
            surface,
            self.selected_dataset.name_for_language(self._language),
            (rect.x + 24, rect.y + 52),
            self._font_small,
            MUTED_TEXT,
        )
        y = rect.y + 78
        for index, model in enumerate(MODELS):
            active = index == self.selected_model_index
            item_rect = pygame.Rect(rect.x + 22, y, rect.width - 44, 62)
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
            train_score = round(self._accuracy_for_model(index, split="train") * 100)
            test_score = round(self._accuracy_for_model(index, split="test") * 100)
            self._draw_text(
                surface,
                f"{train_score}/{test_score}%",
                (item_rect.right - 88, item_rect.y + 12),
                self._font_small,
                model.color if active else MUTED_TEXT,
            )
            self._draw_wrapped(
                surface,
                self._model_card_detail_text(index),
                (item_rect.x + 16, item_rect.y + 39),
                item_rect.width - 32,
                self._font_small,
                MUTED_TEXT,
                line_height=16,
            )
            y += 74
        self._draw_scoreboard(surface, self._side_panel_scoreboard_rect(rect))

    def _draw_scoreboard(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self._draw_text(
            surface,
            self._label("Train / test accuracy", "Train / test accuracy"),
            (rect.x, rect.y),
            self._font_body,
            TEXT,
        )
        y = rect.y + 28
        for index, candidate in enumerate(MODELS):
            train_accuracy = self._accuracy_for_model(index, split="train")
            test_accuracy = self._accuracy_for_model(index, split="test")
            train_rect = pygame.Rect(rect.x + 118, y + 2, round(130 * train_accuracy), 7)
            test_rect = pygame.Rect(rect.x + 118, y + 11, round(130 * test_accuracy), 7)
            label_color = candidate.color if index == self.selected_model_index else MUTED_TEXT
            self._draw_text(
                surface, candidate.name.split()[0], (rect.x, y), self._font_small, label_color
            )
            pygame.draw.rect(surface, GRID, (rect.x + 118, y + 2, 130, 7), border_radius=3)
            pygame.draw.rect(surface, GRID, (rect.x + 118, y + 11, 130, 7), border_radius=3)
            pygame.draw.rect(surface, candidate.color, train_rect, border_radius=3)
            pygame.draw.rect(surface, TEST_POINT_OUTLINE, test_rect, border_radius=3)
            self._draw_text(
                surface,
                f"{round(train_accuracy * 100)}/{round(test_accuracy * 100)}",
                (rect.right - 54, y),
                self._font_small,
                MUTED_TEXT,
            )
            y += 21
        details = (
            f"{self._label('Param', 'Parametr')}: "
            f"{self._complexity_label(self.selected_model_index)}",
            self._test_details_label(self.selected_model_index),
            self._label(
                "E: errors  D: dataset",
                "E: błędy  D: dataset",
            ),
        )
        y += 6
        for line in details:
            self._draw_wrapped(
                surface,
                line,
                (rect.x, y),
                rect.width,
                self._font_small,
                MUTED_TEXT,
                line_height=16,
            )
            y += self._wrapped_height(line, rect.width, self._font_small, line_height=16) + 1

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
            draw_boundary(surface, rect, color, width, self.complexity_levels[index])

    def _draw_logistic_boundary(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        width: int,
        complexity_level: int,
    ) -> None:
        offset = self._logistic_boundary_offset(complexity_level)
        start = self._to_screen((-0.78, 1.05 * 0.78 + offset), rect)
        end = self._to_screen((0.82, -1.05 * 0.82 + offset), rect)
        pygame.draw.line(surface, color, start, end, width)

    def _draw_knn_boundary(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        color: tuple[int, int, int],
        width: int,
        complexity_level: int,
    ) -> None:
        wave = (0.18, 0.34, 0.48)[complexity_level]
        points = tuple(
            self._to_screen(
                (
                    -0.78 + index * 0.18,
                    wave * math.sin(index * 0.9) - 0.16 + index * 0.018,
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
        complexity_level: int,
    ) -> None:
        if complexity_level == 0:
            raw_points = ((-0.72, 0.22), (0.08, 0.22), (0.08, -0.76))
        elif complexity_level == 1:
            raw_points = (
                (-0.72, 0.68),
                (-0.36, 0.68),
                (-0.36, 0.18),
                (0.12, 0.18),
                (0.12, -0.34),
                (0.56, -0.34),
                (0.56, -0.76),
            )
        else:
            raw_points = (
                (-0.78, 0.72),
                (-0.48, 0.72),
                (-0.48, 0.42),
                (-0.18, 0.42),
                (-0.18, 0.02),
                (0.18, 0.02),
                (0.18, -0.28),
                (0.42, -0.28),
                (0.42, -0.58),
                (0.68, -0.58),
                (0.68, -0.82),
            )
        points = tuple(self._to_screen(point, rect) for point in raw_points)
        pygame.draw.lines(surface, color, False, points, width)

    def _draw_points(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        for index, (x, y, class_id) in enumerate(self.points):
            color = CLASS_A if class_id == 0 else CLASS_B
            position = self._to_screen((x, y), rect)
            pygame.draw.circle(surface, color, position, 7)
            pygame.draw.circle(surface, PLOT_BG, position, 7, width=1)
            if self._point_split(index) == "test":
                pygame.draw.circle(surface, TEST_POINT_OUTLINE, position, 10, width=2)
            if self.show_error_highlights and self._is_misclassified(index):
                self._draw_error_marker(surface, position)

    def _draw_plot_legend(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        y = rect.bottom + 18
        labels = (
            (CLASS_A, self._label("class A", "klasa A")),
            (CLASS_B, self._label("class B", "klasa B")),
            (TEST_POINT_OUTLINE, "test"),
            (ERROR_HIGHLIGHT, self._label("error", "błąd")),
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

    def _wrapped_height(
        self,
        text: str,
        width: int,
        font: pygame.font.Font,
        *,
        line_height: int,
    ) -> int:
        lines = 1
        current = ""
        for word in text.split():
            candidate = word if not current else f"{current} {word}"
            if font.size(candidate)[0] <= width:
                current = candidate
                continue

            if current:
                lines += 1
            current = word

        return lines * line_height

    def _side_panel_scoreboard_rect(self, side_panel_rect: pygame.Rect) -> pygame.Rect:
        model_list_start_y = side_panel_rect.y + 78
        model_list_bottom = model_list_start_y + len(MODELS) * 74
        return pygame.Rect(
            side_panel_rect.x + 24,
            model_list_bottom + 6,
            side_panel_rect.width - 48,
            side_panel_rect.bottom - model_list_bottom - 6,
        )

    def _scoreboard_content_bottom_y(self, rect: pygame.Rect) -> int:
        y = rect.y + 28 + len(MODELS) * 21 + 6
        details = (
            f"{self._label('Param', 'Parametr')}: "
            f"{self._complexity_label(self.selected_model_index)}",
            self._test_details_label(self.selected_model_index),
            self._label("E toggles test errors", "E przełącza błędy testowe"),
        )
        for line in details:
            y += self._wrapped_height(line, rect.width, self._font_small, line_height=16) + 1
        return y

    def _draw_error_marker(self, surface: pygame.Surface, position: tuple[int, int]) -> None:
        x, y = position
        pygame.draw.line(surface, ERROR_HIGHLIGHT, (x - 8, y - 8), (x + 8, y + 8), 3)
        pygame.draw.line(surface, ERROR_HIGHLIGHT, (x - 8, y + 8), (x + 8, y - 8), 3)

    def _model_card_detail_text(self, model_index: int) -> str:
        shape_labels = (
            self._label("linear", "linear"),
            self._label("local", "local"),
            self._label("rules", "rules"),
        )
        return f"{self._complexity_label(model_index)} · {shape_labels[model_index]}"

    def _model_card_detail_bottom_y(self, item_rect: pygame.Rect, model_index: int) -> int:
        return (
            item_rect.y
            + 39
            + self._wrapped_height(
                self._model_card_detail_text(model_index),
                item_rect.width - 32,
                self._font_small,
                line_height=16,
            )
        )

    def _to_screen(self, point: tuple[float, float], rect: pygame.Rect) -> tuple[int, int]:
        x, y = point
        return (
            rect.left + round((x + 1.0) * 0.5 * rect.width),
            rect.top + round((1.0 - (y + 1.0) * 0.5) * rect.height),
        )

    def _accuracy_for_model(self, model_index: int, *, split: str | None = None) -> float:
        correct = 0
        total = 0
        for sample_index, point in enumerate(self.points):
            if split is not None and self._point_split(sample_index) != split:
                continue

            prediction = self._predict_model(model_index, point, sample_index)
            if prediction == point[2]:
                correct += 1
            total += 1

        return correct / total

    def _is_misclassified(self, sample_index: int) -> bool:
        if self._point_split(sample_index) != "test":
            return False

        point = self.points[sample_index]
        prediction = self._predict_model(self.selected_model_index, point, sample_index)
        return prediction != point[2]

    def _confusion_counts(
        self,
        model_index: int,
        *,
        split: str | None = None,
    ) -> dict[str, int]:
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for sample_index, point in enumerate(self.points):
            if split is not None and self._point_split(sample_index) != split:
                continue

            prediction = self._predict_model(model_index, point, sample_index)
            actual = point[2]
            if prediction == 1 and actual == 1:
                counts["tp"] += 1
            elif prediction == 1 and actual == 0:
                counts["fp"] += 1
            elif prediction == 0 and actual == 1:
                counts["fn"] += 1
            else:
                counts["tn"] += 1

        return counts

    def _precision_recall_for_model(
        self,
        model_index: int,
        *,
        split: str | None = None,
    ) -> tuple[float, float]:
        counts = self._confusion_counts(model_index, split=split)
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        return (precision, recall)

    def _confusion_summary_label(self, model_index: int, *, split: str) -> str:
        counts = self._confusion_counts(model_index, split=split)
        return f"{split}: TP {counts['tp']} FP {counts['fp']} FN {counts['fn']} TN {counts['tn']}"

    def _precision_recall_label(self, model_index: int, *, split: str) -> str:
        precision, recall = self._precision_recall_for_model(model_index, split=split)
        return f"precision/recall {round(precision * 100)}/{round(recall * 100)}"

    def _test_details_label(self, model_index: int) -> str:
        counts = self._confusion_counts(model_index, split="test")
        precision, recall = self._precision_recall_for_model(model_index, split="test")
        return (
            f"test TP/FP/FN/TN {counts['tp']}/{counts['fp']}/{counts['fn']}/{counts['tn']} "
            f"P/R {round(precision * 100)}/{round(recall * 100)}"
        )

    def _point_split(self, sample_index: int) -> str:
        if sample_index % 4 == 3:
            return "test"

        return "train"

    def _predict_model(
        self,
        model_index: int,
        point: tuple[float, float, int],
        sample_index: int,
    ) -> int:
        x, y, _class_id = point
        if model_index == 0:
            offset = self._logistic_boundary_offset(self.complexity_levels[model_index])
            return int(y + 1.05 * x - offset > 0.0)
        if model_index == 1:
            return self._predict_knn(point, sample_index, self.complexity_levels[model_index])

        return self._predict_tree(point, self.complexity_levels[model_index])

    def _predict_knn(
        self,
        point: tuple[float, float, int],
        sample_index: int,
        complexity_level: int,
    ) -> int:
        x, y, _class_id = point
        neighbors: list[tuple[float, int]] = []
        for index, candidate in enumerate(self.points):
            if index == sample_index:
                continue
            candidate_x, candidate_y, candidate_class = candidate
            distance = (candidate_x - x) ** 2 + (candidate_y - y) ** 2
            neighbors.append((distance, candidate_class))

        k = KNN_K_VALUES[complexity_level]
        nearest = sorted(neighbors)[:k]
        votes = sum(class_id for _distance, class_id in nearest)
        return int(votes >= math.ceil(k / 2))

    def _predict_tree(self, point: tuple[float, float, int], complexity_level: int) -> int:
        x, y, _class_id = point
        if complexity_level == 0:
            return int(y > 0.22 or x > 0.08)
        if complexity_level == 1:
            return int((y > 0.55 and x > -0.55) or (x > 0.12 and y > -0.42) or x > 0.55)

        return int(
            (y > 0.62 and x > -0.58)
            or (-0.18 < x <= 0.18 and y > 0.02)
            or (x > 0.18 and y > -0.28)
            or (x > 0.42 and y > -0.58)
            or x > 0.68
        )

    def _logistic_boundary_offset(self, complexity_level: int) -> float:
        threshold = LOGISTIC_THRESHOLDS[complexity_level]
        return (threshold - 0.50) * 1.2 - 0.09

    def _complexity_label(self, model_index: int) -> str:
        complexity_level = self.complexity_levels[model_index]
        if model_index == 0:
            return f"threshold={LOGISTIC_THRESHOLDS[complexity_level]:.2f}"
        if model_index == 1:
            return f"k={KNN_K_VALUES[complexity_level]}"

        return f"depth={TREE_DEPTHS[complexity_level]}"

    def _label(self, en: str, pl: str) -> str:
        if self._language == "pl":
            return pl

        return en


def create_model_comparison_lab_scene(context: AppContext) -> ModelComparisonLabScene:
    """Create the unified shell Model Comparison Lab scene."""
    return ModelComparisonLabScene(context)
