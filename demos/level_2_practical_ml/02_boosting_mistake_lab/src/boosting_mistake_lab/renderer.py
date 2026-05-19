"""Pygame renderer for the Boosting Mistake Lab demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import pygame
from numpy.typing import NDArray

from boosting_mistake_lab.boosted_prediction import predict_boosted_ensemble
from boosting_mistake_lab.dataset import WeightedTrainTestDataset
from boosting_mistake_lab.trainer import BoostingTrainerResult
from boosting_mistake_lab.weak_learner import WeakLearnerBaseline

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type BoolArray = NDArray[np.bool_]
type Predictor = Callable[[FloatArray], tuple[int, float]]

WINDOW_WIDTH: Final[int] = 1320
WINDOW_HEIGHT: Final[int] = 780
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)

BACKGROUND_COLOR: Final[tuple[int, int, int]] = (245, 247, 250)
PANEL_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
TEXT_COLOR: Final[tuple[int, int, int]] = (30, 35, 40)
MUTED_TEXT_COLOR: Final[tuple[int, int, int]] = (100, 110, 120)
CLASS_ZERO_COLOR: Final[tuple[int, int, int]] = (60, 130, 230)
CLASS_ONE_COLOR: Final[tuple[int, int, int]] = (235, 130, 60)
CLASS_ZERO_REGION_COLOR: Final[tuple[int, int, int]] = (226, 238, 255)
CLASS_ONE_REGION_COLOR: Final[tuple[int, int, int]] = (255, 235, 220)
TRAIN_POINT_BORDER_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
TEST_POINT_BORDER_COLOR: Final[tuple[int, int, int]] = (35, 35, 35)
AXIS_COLOR: Final[tuple[int, int, int]] = (180, 185, 190)
ERROR_COLOR: Final[tuple[int, int, int]] = (190, 60, 60)
SUCCESS_COLOR: Final[tuple[int, int, int]] = (35, 135, 85)

LEFT_PLOT_RECT: Final[pygame.Rect] = pygame.Rect(40, 40, 430, 520)
RIGHT_PLOT_RECT: Final[pygame.Rect] = pygame.Rect(500, 40, 430, 520)
SIDE_RECT: Final[pygame.Rect] = pygame.Rect(960, 40, 320, 540)
BOTTOM_RECT: Final[pygame.Rect] = pygame.Rect(40, 600, 1240, 120)

PADDING: Final[int] = 34
PANEL_RADIUS: Final[int] = 14
BASE_POINT_RADIUS: Final[int] = 5
TEST_POINT_SIZE: Final[int] = 9
AXIS_WIDTH: Final[int] = 1
ERROR_MARK_WIDTH: Final[int] = 2
GRID_CELL_SIZE: Final[int] = 16
TEXT_LINE_HEIGHT: Final[int] = 28
SMALL_TEXT_LINE_HEIGHT: Final[int] = 21

FEATURE_X_INDEX: Final[int] = 0
FEATURE_Y_INDEX: Final[int] = 1
CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1

MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.18
LOW_CONFIDENCE_BASELINE: Final[float] = 0.50
LOW_CONFIDENCE_ALPHA: Final[float] = 0.30
HIGH_CONFIDENCE_ALPHA: Final[float] = 0.95

CLASS_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    CLASS_ZERO_LABEL: CLASS_ZERO_COLOR,
    CLASS_ONE_LABEL: CLASS_ONE_COLOR,
}
REGION_COLORS: Final[dict[int, tuple[int, int, int]]] = {
    CLASS_ZERO_LABEL: CLASS_ZERO_REGION_COLOR,
    CLASS_ONE_LABEL: CLASS_ONE_REGION_COLOR,
}


@dataclass(frozen=True, slots=True)
class WorldBounds:
    """Numeric plot bounds shared by both model panels."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True, slots=True)
class BoostingRenderState:
    """State required to render one Boosting Mistake Lab frame."""

    dataset: WeightedTrainTestDataset
    trainer_result: BoostingTrainerResult
    dataset_kind: str
    noise_std: float
    seed: int
    round_count: int
    min_samples_leaf: int
    confidence_view_enabled: bool


class BoostingRenderer:
    """Render boosting training and ensemble behavior using Pygame."""

    def __init__(self, screen: pygame.Surface) -> None:
        """Initialize renderer resources."""
        self._screen = screen
        self._font = pygame.font.Font(None, 28)
        self._small_font = pygame.font.Font(None, 22)
        self._title_font = pygame.font.Font(None, 34)

    def draw(self, state: BoostingRenderState) -> None:
        """Draw the full UI frame."""
        self._screen.fill(BACKGROUND_COLOR)

        self._draw_panel(LEFT_PLOT_RECT)
        self._draw_panel(RIGHT_PLOT_RECT)
        self._draw_panel(SIDE_RECT)
        self._draw_panel(BOTTOM_RECT)

        train_features = np.asarray(state.dataset.train.snapshot.features, dtype=float)
        test_features = np.asarray(state.dataset.test.snapshot.features, dtype=float)
        bounds = _compute_world_bounds(train_features, test_features)

        final_round = state.trainer_result.round_results[-1]

        self._draw_weak_learner_panel(
            state=state,
            rect=LEFT_PLOT_RECT,
            bounds=bounds,
            weak_learner=final_round.weak_learner,
        )
        self._draw_boosted_panel(
            state=state,
            rect=RIGHT_PLOT_RECT,
            bounds=bounds,
        )
        self._draw_side_panel(state)
        self._draw_bottom_panel()

        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw one rounded panel."""
        pygame.draw.rect(self._screen, PANEL_COLOR, rect, border_radius=PANEL_RADIUS)

    def _draw_weak_learner_panel(
        self,
        *,
        state: BoostingRenderState,
        rect: pygame.Rect,
        bounds: WorldBounds,
        weak_learner: WeakLearnerBaseline,
    ) -> None:
        """Draw the last weak learner from the trainer."""
        weak_snapshot = state.trainer_result.round_results[-1].weak_snapshot
        train_features = np.asarray(weak_snapshot.visual_state["train_features"], dtype=float)
        train_targets = np.asarray(weak_snapshot.visual_state["train_targets"], dtype=int)
        train_weights = np.asarray(
            weak_snapshot.visual_state["train_sample_weights"],
            dtype=float,
        )
        test_features = np.asarray(weak_snapshot.visual_state["test_features"], dtype=float)
        test_targets = np.asarray(weak_snapshot.visual_state["test_targets"], dtype=int)
        test_predictions = np.asarray(
            weak_snapshot.visual_state["test_predictions"],
            dtype=int,
        )

        self._draw_decision_regions(
            rect=rect,
            bounds=bounds,
            predictor=lambda point: _predict_weak(weak_learner, point),
            confidence_view_enabled=False,
        )
        self._draw_axes(rect, bounds)
        self._draw_train_points(
            rect=rect,
            features=train_features,
            targets=train_targets,
            sample_weights=train_weights,
            bounds=bounds,
        )
        self._draw_test_points(
            rect=rect,
            features=test_features,
            targets=test_targets,
            predictions=test_predictions,
            bounds=bounds,
        )
        self._draw_text(
            f"Weak learner — round {state.round_count}",
            rect.left + PADDING,
            rect.top + 16,
            self._font,
            TEXT_COLOR,
        )

    def _draw_boosted_panel(
        self,
        *,
        state: BoostingRenderState,
        rect: pygame.Rect,
        bounds: WorldBounds,
    ) -> None:
        """Draw the boosted ensemble."""
        train_features = np.asarray(state.dataset.train.snapshot.features, dtype=float)
        train_targets = np.asarray(state.dataset.train.snapshot.targets, dtype=int)
        train_weights = np.asarray(state.trainer_result.final_train_weights, dtype=float)
        test_features = np.asarray(state.dataset.test.snapshot.features, dtype=float)
        test_targets = np.asarray(state.dataset.test.snapshot.targets, dtype=int)
        test_predictions = np.asarray(
            state.trainer_result.snapshot.visual_state["boosted_test_predictions"],
            dtype=int,
        )

        self._draw_decision_regions(
            rect=rect,
            bounds=bounds,
            predictor=lambda point: _predict_boosted(state, point),
            confidence_view_enabled=state.confidence_view_enabled,
        )
        self._draw_axes(rect, bounds)
        self._draw_train_points(
            rect=rect,
            features=train_features,
            targets=train_targets,
            sample_weights=train_weights,
            bounds=bounds,
        )
        self._draw_test_points(
            rect=rect,
            features=test_features,
            targets=test_targets,
            predictions=test_predictions,
            bounds=bounds,
        )
        self._draw_text(
            f"Boosted ensemble — {state.round_count} rounds",
            rect.left + PADDING,
            rect.top + 16,
            self._font,
            TEXT_COLOR,
        )

    def _draw_decision_regions(
        self,
        *,
        rect: pygame.Rect,
        bounds: WorldBounds,
        predictor: Predictor,
        confidence_view_enabled: bool,
    ) -> None:
        """Draw approximate decision regions using a coarse grid."""
        drawable = _drawable_rect(rect)

        for screen_x in range(drawable.left, drawable.right, GRID_CELL_SIZE):
            for screen_y in range(drawable.top, drawable.bottom, GRID_CELL_SIZE):
                world_x, world_y = _screen_to_world(screen_x, screen_y, bounds, rect)
                point = np.array([[world_x, world_y]], dtype=float)
                prediction, confidence = predictor(point)
                color = _region_color(
                    prediction=prediction,
                    confidence=confidence,
                    confidence_view_enabled=confidence_view_enabled,
                )
                cell = pygame.Rect(screen_x, screen_y, GRID_CELL_SIZE, GRID_CELL_SIZE)
                pygame.draw.rect(self._screen, color, cell)

    def _draw_axes(self, rect: pygame.Rect, bounds: WorldBounds) -> None:
        """Draw x=0 and y=0 axes when visible."""
        if bounds.y_min <= 0.0 <= bounds.y_max:
            start = _world_to_screen(bounds.x_min, 0.0, bounds, rect)
            end = _world_to_screen(bounds.x_max, 0.0, bounds, rect)
            pygame.draw.line(self._screen, AXIS_COLOR, start, end, AXIS_WIDTH)

        if bounds.x_min <= 0.0 <= bounds.x_max:
            start = _world_to_screen(0.0, bounds.y_min, bounds, rect)
            end = _world_to_screen(0.0, bounds.y_max, bounds, rect)
            pygame.draw.line(self._screen, AXIS_COLOR, start, end, AXIS_WIDTH)

    def _draw_train_points(
        self,
        *,
        rect: pygame.Rect,
        features: FloatArray,
        targets: IntArray,
        sample_weights: FloatArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw training points as circles scaled by sample weight."""
        sample_count = features.shape[0]

        for point, target, weight in zip(features, targets, sample_weights, strict=True):
            position = _world_to_screen(float(point[0]), float(point[1]), bounds, rect)
            radius = _weight_to_radius(float(weight), sample_count)
            color = CLASS_COLORS.get(int(target), MUTED_TEXT_COLOR)

            pygame.draw.circle(self._screen, TRAIN_POINT_BORDER_COLOR, position, radius + 2)
            pygame.draw.circle(self._screen, color, position, radius)

    def _draw_test_points(
        self,
        *,
        rect: pygame.Rect,
        features: FloatArray,
        targets: IntArray,
        predictions: IntArray,
        bounds: WorldBounds,
    ) -> None:
        """Draw test points as outlined squares and mark mistakes."""
        for point, target, prediction in zip(features, targets, predictions, strict=True):
            position = _world_to_screen(float(point[0]), float(point[1]), bounds, rect)
            color = CLASS_COLORS.get(int(target), MUTED_TEXT_COLOR)
            square = pygame.Rect(0, 0, TEST_POINT_SIZE, TEST_POINT_SIZE)
            square.center = position

            pygame.draw.rect(self._screen, color, square)
            pygame.draw.rect(self._screen, TEST_POINT_BORDER_COLOR, square, width=1)

            if int(target) != int(prediction):
                self._draw_error_mark(position)

    def _draw_error_mark(self, position: tuple[int, int]) -> None:
        """Draw an X mark for a misclassified test point."""
        x, y = position
        offset = BASE_POINT_RADIUS + 5

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

    def _draw_side_panel(self, state: BoostingRenderState) -> None:
        """Draw metrics and current configuration."""
        metrics = state.trainer_result.snapshot.metrics
        x = SIDE_RECT.left + 24
        y = SIDE_RECT.top + 20

        self._draw_text("Boosting", x, y, self._title_font, TEXT_COLOR)
        y += 38

        rows = [
            ("Dataset", state.dataset_kind),
            ("Rounds", str(state.round_count)),
            ("Min leaf", str(state.min_samples_leaf)),
            ("Noise", f"{state.noise_std:.2f}"),
            ("Seed", str(state.seed)),
            ("Conf. view", _enabled_text(state.confidence_view_enabled)),
            ("Weak train", f"{metrics['final_train_accuracy']:.3f}"),
            ("Weak test", f"{metrics['final_test_accuracy']:.3f}"),
            ("Boost train", f"{metrics['boosted_train_accuracy']:.3f}"),
            ("Boost test", f"{metrics['boosted_test_accuracy']:.3f}"),
            ("Gap", f"{metrics['boosted_generalization_gap']:.3f}"),
            ("Confidence", f"{metrics['mean_boosted_test_confidence']:.3f}"),
            ("Best round", str(int(metrics["best_staged_round_index"]))),
            ("Best test", f"{metrics['best_staged_boosted_test_accuracy']:.3f}"),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 126, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        y += 8
        self._draw_staged_accuracy_summary(state, x, y)

    def _draw_staged_accuracy_summary(
        self,
        state: BoostingRenderState,
        x: int,
        y: int,
    ) -> None:
        """Draw compact staged boosted accuracy summary."""
        history = state.trainer_result.staged_history

        self._draw_text("Staged boosted test acc", x, y, self._font, TEXT_COLOR)
        y += 28

        for round_index, test_accuracy in zip(
            history.round_indices,
            history.boosted_test_accuracies,
            strict=True,
        ):
            label = f"r{int(round_index)}"
            value = f"{float(test_accuracy):.3f}"
            self._draw_text(label, x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(value, x + 48, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

    def _draw_bottom_panel(self) -> None:
        """Draw controls and legend."""
        x = BOTTOM_RECT.left + 24
        y = BOTTOM_RECT.top + 14

        controls = (
            "D: dataset   Up/Down: rounds   W/S: min leaf   "
            "C: confidence view   Left/Right: noise   N: seed   R: reset"
        )
        self._draw_text(controls, x, y, self._small_font, TEXT_COLOR)

        legend = (
            "Circles = train samples, size = current/final sample weight. "
            "Squares = test samples, X = misclassified test samples."
        )
        self._draw_text(legend, x, y + TEXT_LINE_HEIGHT, self._small_font, MUTED_TEXT_COLOR)

        explanation = (
            "Left panel shows the last weak learner. Right panel shows alpha-weighted "
            "boosted ensemble predictions after the selected number of rounds."
        )
        self._draw_text(
            explanation,
            x,
            y + 2 * TEXT_LINE_HEIGHT,
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
        """Draw text on screen."""
        surface = font.render(text, True, color)
        self._screen.blit(surface, (x, y))


def _predict_weak(
    weak_learner: WeakLearnerBaseline,
    point: FloatArray,
) -> tuple[int, float]:
    """Predict one point using a weak learner."""
    return int(weak_learner.predict(point)[0]), 1.0


def _predict_boosted(
    state: BoostingRenderState,
    point: FloatArray,
) -> tuple[int, float]:
    """Predict one point using the boosted ensemble."""
    result = predict_boosted_ensemble(
        weak_learners=[
            round_result.weak_learner for round_result in state.trainer_result.round_results
        ],
        learner_weights=state.trainer_result.learner_weights,
        features=point,
    )

    return int(result.predictions[0]), float(result.confidence[0])


def _region_color(
    *,
    prediction: int,
    confidence: float,
    confidence_view_enabled: bool,
) -> tuple[int, int, int]:
    """Return region color, optionally faded according to confidence."""
    base_color = REGION_COLORS.get(prediction, BACKGROUND_COLOR)

    if not confidence_view_enabled:
        return base_color

    alpha = _confidence_to_alpha(confidence)

    return _blend_colors(PANEL_COLOR, base_color, alpha)


def _confidence_to_alpha(confidence: float) -> float:
    """Convert confidence into region color intensity."""
    clamped_confidence = min(1.0, max(LOW_CONFIDENCE_BASELINE, confidence))
    normalized = (clamped_confidence - LOW_CONFIDENCE_BASELINE) / (1.0 - LOW_CONFIDENCE_BASELINE)

    return LOW_CONFIDENCE_ALPHA + normalized * (HIGH_CONFIDENCE_ALPHA - LOW_CONFIDENCE_ALPHA)


def _blend_colors(
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    """Blend foreground color over background color."""
    return tuple(
        round(background_channel * (1.0 - alpha) + foreground_channel * alpha)
        for background_channel, foreground_channel in zip(background, foreground, strict=True)
    )


def _enabled_text(value: bool) -> str:
    """Return short enabled/disabled text."""
    if value:
        return "on"

    return "off"


def _weight_to_radius(weight: float, sample_count: int) -> int:
    """Convert normalized sample weight to point radius."""
    relative_weight = max(0.0, weight * sample_count)

    return round(4.0 + min(10.0, 3.0 * np.sqrt(relative_weight)))


def _compute_world_bounds(train_features: FloatArray, test_features: FloatArray) -> WorldBounds:
    """Compute shared bounds for train and test data."""
    features = np.vstack([train_features, test_features])

    x_min = float(np.min(features[:, FEATURE_X_INDEX]))
    x_max = float(np.max(features[:, FEATURE_X_INDEX]))
    y_min = float(np.min(features[:, FEATURE_Y_INDEX]))
    y_max = float(np.max(features[:, FEATURE_Y_INDEX]))

    x_span = max(x_max - x_min, MIN_WORLD_SPAN)
    y_span = max(y_max - y_min, MIN_WORLD_SPAN)

    return WorldBounds(
        x_min=x_min - x_span * WORLD_MARGIN_RATIO,
        x_max=x_max + x_span * WORLD_MARGIN_RATIO,
        y_min=y_min - y_span * WORLD_MARGIN_RATIO,
        y_max=y_max + y_span * WORLD_MARGIN_RATIO,
    )


def _drawable_rect(rect: pygame.Rect) -> pygame.Rect:
    """Return inner drawable plot area."""
    return pygame.Rect(
        rect.left + PADDING,
        rect.top + PADDING,
        rect.width - 2 * PADDING,
        rect.height - 2 * PADDING,
    )


def _world_to_screen(
    x_value: float,
    y_value: float,
    bounds: WorldBounds,
    rect: pygame.Rect,
) -> tuple[int, int]:
    """Convert world coordinates to screen coordinates."""
    drawable = _drawable_rect(rect)
    x_ratio = (x_value - bounds.x_min) / (bounds.x_max - bounds.x_min)
    y_ratio = (y_value - bounds.y_min) / (bounds.y_max - bounds.y_min)

    screen_x = drawable.left + int(x_ratio * drawable.width)
    screen_y = drawable.bottom - int(y_ratio * drawable.height)

    return screen_x, screen_y


def _screen_to_world(
    screen_x: int,
    screen_y: int,
    bounds: WorldBounds,
    rect: pygame.Rect,
) -> tuple[float, float]:
    """Convert screen coordinates to world coordinates."""
    drawable = _drawable_rect(rect)
    x_ratio = (screen_x - drawable.left) / drawable.width
    y_ratio = (drawable.bottom - screen_y) / drawable.height

    world_x = bounds.x_min + x_ratio * (bounds.x_max - bounds.x_min)
    world_y = bounds.y_min + y_ratio * (bounds.y_max - bounds.y_min)

    return world_x, world_y
