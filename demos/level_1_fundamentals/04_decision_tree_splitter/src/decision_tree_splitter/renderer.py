"""Pygame renderer for the Decision Tree Splitter demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pygame
from ml_lab_core import AlgorithmSnapshot
from numpy.typing import NDArray

from decision_tree_splitter.challenge import DecisionTreeChallengeResult
from decision_tree_splitter.split import SplitCandidate
from decision_tree_splitter.tree import DecisionTreeNode

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int_]
type BoolArray = NDArray[np.bool_]

WINDOW_WIDTH: Final[int] = 1120
WINDOW_HEIGHT: Final[int] = 740
WINDOW_SIZE: Final[tuple[int, int]] = (WINDOW_WIDTH, WINDOW_HEIGHT)

MODE_AUTO_TREE: Final[str] = "auto_tree"
MODE_MANUAL_SPLIT: Final[str] = "manual_split"

BACKGROUND_COLOR: Final[tuple[int, int, int]] = (245, 247, 250)
PANEL_COLOR: Final[tuple[int, int, int]] = (255, 255, 255)
TEXT_COLOR: Final[tuple[int, int, int]] = (30, 35, 40)
MUTED_TEXT_COLOR: Final[tuple[int, int, int]] = (100, 110, 120)
CLASS_ZERO_COLOR: Final[tuple[int, int, int]] = (60, 130, 230)
CLASS_ONE_COLOR: Final[tuple[int, int, int]] = (235, 130, 60)
CLASS_ZERO_REGION_COLOR: Final[tuple[int, int, int]] = (226, 238, 255)
CLASS_ONE_REGION_COLOR: Final[tuple[int, int, int]] = (255, 235, 220)
SPLIT_COLOR: Final[tuple[int, int, int]] = (35, 35, 35)
MANUAL_SPLIT_COLOR: Final[tuple[int, int, int]] = (110, 70, 190)
AXIS_COLOR: Final[tuple[int, int, int]] = (180, 185, 190)
ERROR_COLOR: Final[tuple[int, int, int]] = (190, 60, 60)

MAIN_RECT: Final[pygame.Rect] = pygame.Rect(40, 40, 700, 540)
SIDE_RECT: Final[pygame.Rect] = pygame.Rect(780, 40, 300, 540)
BOTTOM_RECT: Final[pygame.Rect] = pygame.Rect(40, 610, 1040, 90)

PADDING: Final[int] = 36
POINT_RADIUS: Final[int] = 6
SPLIT_WIDTH: Final[int] = 3
AXIS_WIDTH: Final[int] = 1
ERROR_MARK_WIDTH: Final[int] = 2
PANEL_RADIUS: Final[int] = 14
TEXT_LINE_HEIGHT: Final[int] = 28
SMALL_TEXT_LINE_HEIGHT: Final[int] = 22

MIN_WORLD_SPAN: Final[float] = 1.0
WORLD_MARGIN_RATIO: Final[float] = 0.18
FEATURE_X_INDEX: Final[int] = 0
FEATURE_Y_INDEX: Final[int] = 1
CLASS_ZERO_LABEL: Final[int] = 0
CLASS_ONE_LABEL: Final[int] = 1

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
    """Numeric coordinate bounds used to map data points onto the screen."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class DecisionTreeRenderer:
    """Render decision tree state using Pygame."""

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
        dataset_kind: str,
        noise_std: float,
        seed: int,
        mode: str,
        manual_snapshot: AlgorithmSnapshot | None,
        manual_error: str | None,
        manual_feature_index: int,
        manual_threshold: float,
        challenge_result: DecisionTreeChallengeResult,
    ) -> None:
        """Draw the full frame.

        Args:
            snapshot: Current recursive decision tree snapshot.
            dataset_kind: Current synthetic dataset kind.
            noise_std: Current dataset noise standard deviation.
            seed: Current dataset seed.
            mode: Current UI mode.
            manual_snapshot: Manual split snapshot if valid.
            manual_error: Manual split error if the selected split is invalid.
            manual_feature_index: Current manual split feature.
            manual_threshold: Current manual split threshold.
            challenge_result: Current challenge status.
        """
        self._screen.fill(BACKGROUND_COLOR)

        self._draw_panel(MAIN_RECT)
        self._draw_panel(SIDE_RECT)
        self._draw_panel(BOTTOM_RECT)

        if mode == MODE_MANUAL_SPLIT:
            self._draw_manual_plot(snapshot, manual_snapshot, manual_error)
        else:
            self._draw_tree_plot(snapshot)

        self._draw_side_panel(
            snapshot,
            dataset_kind=dataset_kind,
            noise_std=noise_std,
            seed=seed,
            mode=mode,
            manual_snapshot=manual_snapshot,
            manual_error=manual_error,
            manual_feature_index=manual_feature_index,
            manual_threshold=manual_threshold,
            challenge_result=challenge_result,
        )
        self._draw_bottom_panel(
            snapshot,
            mode=mode,
            manual_error=manual_error,
            challenge_result=challenge_result,
        )

        pygame.display.flip()

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw a rounded white panel."""
        pygame.draw.rect(
            self._screen,
            PANEL_COLOR,
            rect,
            border_radius=PANEL_RADIUS,
        )

    def _draw_tree_plot(self, snapshot: AlgorithmSnapshot) -> None:
        """Draw tree regions, split lines, and data points."""
        features = np.asarray(snapshot.visual_state["features"], dtype=float)
        targets = np.asarray(snapshot.visual_state["targets"], dtype=int)
        predictions = np.asarray(snapshot.visual_state["predictions"], dtype=int)
        root = _extract_root(snapshot)

        bounds = _compute_world_bounds(features)

        self._draw_leaf_regions(root, bounds, bounds)
        self._draw_axes(bounds)
        self._draw_split_lines(root, bounds, bounds)
        self._draw_points(features, targets, predictions, bounds)

        self._draw_text(
            "Automatic recursive tree",
            MAIN_RECT.left + PADDING,
            MAIN_RECT.top + 18,
            self._font,
            TEXT_COLOR,
        )

    def _draw_manual_plot(
        self,
        tree_snapshot: AlgorithmSnapshot,
        manual_snapshot: AlgorithmSnapshot | None,
        manual_error: str | None,
    ) -> None:
        """Draw manual split regions and data points."""
        features = np.asarray(tree_snapshot.visual_state["features"], dtype=float)
        targets = np.asarray(tree_snapshot.visual_state["targets"], dtype=int)
        bounds = _compute_world_bounds(features)

        if manual_snapshot is None:
            predictions = np.asarray(tree_snapshot.visual_state["predictions"], dtype=int)

            self._draw_axes(bounds)
            self._draw_points(features, targets, predictions, bounds)
            self._draw_manual_title(manual_error)
            return

        candidate = manual_snapshot.visual_state["candidate"]

        if not isinstance(candidate, SplitCandidate):
            msg = "manual_snapshot.visual_state['candidate'] must be a SplitCandidate."
            raise TypeError(msg)

        left_mask = np.asarray(manual_snapshot.visual_state["left_mask"], dtype=bool)
        right_mask = np.asarray(manual_snapshot.visual_state["right_mask"], dtype=bool)

        left_prediction = _majority_prediction(targets[left_mask])
        right_prediction = _majority_prediction(targets[right_mask])
        predictions = _manual_predictions(
            targets=targets,
            left_mask=left_mask,
            right_mask=right_mask,
            left_prediction=left_prediction,
            right_prediction=right_prediction,
        )

        left_bounds, right_bounds = _split_bounds(bounds, candidate)

        self._draw_region(left_bounds, bounds, left_prediction)
        self._draw_region(right_bounds, bounds, right_prediction)
        self._draw_axes(bounds)
        self._draw_one_split_line(
            candidate,
            bounds,
            bounds,
            color=MANUAL_SPLIT_COLOR,
        )
        self._draw_points(features, targets, predictions, bounds)
        self._draw_manual_title(manual_error)

    def _draw_manual_title(self, manual_error: str | None) -> None:
        """Draw manual split plot title."""
        title = "Manual split mode"

        if manual_error is not None:
            title = "Manual split mode — invalid split"

        self._draw_text(
            title,
            MAIN_RECT.left + PADDING,
            MAIN_RECT.top + 18,
            self._font,
            TEXT_COLOR,
        )

    def _draw_leaf_regions(
        self,
        node: DecisionTreeNode,
        node_bounds: WorldBounds,
        global_bounds: WorldBounds,
    ) -> None:
        """Draw colored rectangular regions for leaf predictions."""
        if node.is_leaf:
            self._draw_region(node_bounds, global_bounds, node.prediction)
            return

        assert node.split_evaluation is not None
        assert node.left is not None
        assert node.right is not None

        left_bounds, right_bounds = _split_bounds(
            node_bounds,
            node.split_evaluation.candidate,
        )

        self._draw_leaf_regions(node.left, left_bounds, global_bounds)
        self._draw_leaf_regions(node.right, right_bounds, global_bounds)

    def _draw_region(
        self,
        region_bounds: WorldBounds,
        global_bounds: WorldBounds,
        prediction: int,
    ) -> None:
        """Draw one colored decision region."""
        color = REGION_COLORS.get(prediction, BACKGROUND_COLOR)
        rect = _world_bounds_to_screen_rect(region_bounds, global_bounds, MAIN_RECT)

        pygame.draw.rect(self._screen, color, rect)

    def _draw_split_lines(
        self,
        node: DecisionTreeNode,
        node_bounds: WorldBounds,
        global_bounds: WorldBounds,
    ) -> None:
        """Draw split lines recursively."""
        if node.is_leaf:
            return

        assert node.split_evaluation is not None
        assert node.left is not None
        assert node.right is not None

        candidate = node.split_evaluation.candidate
        self._draw_one_split_line(candidate, node_bounds, global_bounds, color=SPLIT_COLOR)

        left_bounds, right_bounds = _split_bounds(node_bounds, candidate)

        self._draw_split_lines(node.left, left_bounds, global_bounds)
        self._draw_split_lines(node.right, right_bounds, global_bounds)

    def _draw_one_split_line(
        self,
        candidate: SplitCandidate,
        node_bounds: WorldBounds,
        global_bounds: WorldBounds,
        *,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one vertical or horizontal split line."""
        threshold = candidate.threshold

        if candidate.feature_index == FEATURE_X_INDEX:
            start = _world_to_screen(threshold, node_bounds.y_min, global_bounds, MAIN_RECT)
            end = _world_to_screen(threshold, node_bounds.y_max, global_bounds, MAIN_RECT)
        else:
            start = _world_to_screen(node_bounds.x_min, threshold, global_bounds, MAIN_RECT)
            end = _world_to_screen(node_bounds.x_max, threshold, global_bounds, MAIN_RECT)

        pygame.draw.line(self._screen, color, start, end, SPLIT_WIDTH)

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
        dataset_kind: str,
        noise_std: float,
        seed: int,
        mode: str,
        manual_snapshot: AlgorithmSnapshot | None,
        manual_error: str | None,
        manual_feature_index: int,
        manual_threshold: float,
        challenge_result: DecisionTreeChallengeResult,
    ) -> None:
        """Draw metrics and current controls."""
        x = SIDE_RECT.left + 24
        y = SIDE_RECT.top + 22

        self._draw_text("Decision tree", x, y, self._title_font, TEXT_COLOR)
        y += 44

        rows = [
            ("Mode", mode),
            ("Dataset", dataset_kind),
            ("Criterion", str(snapshot.metrics["criterion"])),
            ("Max depth", f"{float(snapshot.metrics['max_depth']):.0f}"),
            ("Actual depth", f"{float(snapshot.metrics['actual_depth']):.0f}"),
            ("Accuracy", f"{float(snapshot.metrics['training_accuracy']):.3f}"),
            ("Nodes", f"{float(snapshot.metrics['node_count']):.0f}"),
            ("Leaves", f"{float(snapshot.metrics['leaf_count']):.0f}"),
            ("Noise", f"{noise_std:.2f}"),
            ("Seed", str(seed)),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(str(value), x + 130, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        y += 8
        self._draw_challenge_panel(challenge_result, x, y)
        y += 94

        if mode == MODE_MANUAL_SPLIT:
            self._draw_manual_split_panel(
                manual_snapshot,
                manual_error=manual_error,
                feature_index=manual_feature_index,
                threshold=manual_threshold,
                x=x,
                y=y,
            )
        else:
            self._draw_root_split(snapshot, x, y)

    def _draw_challenge_panel(
        self,
        challenge_result: DecisionTreeChallengeResult,
        x: int,
        y: int,
    ) -> None:
        """Draw challenge status and targets."""
        self._draw_text("Challenge", x, y, self._font, TEXT_COLOR)
        y += 26

        rows = [
            ("Status", challenge_result.status),
            (
                "Accuracy",
                f"{challenge_result.accuracy:.2f}/{challenge_result.target_accuracy:.2f}",
            ),
            (
                "Depth",
                f"{challenge_result.max_depth}/{challenge_result.max_allowed_depth}",
            ),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(value, x + 100, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

    def _draw_manual_split_panel(
        self,
        manual_snapshot: AlgorithmSnapshot | None,
        *,
        manual_error: str | None,
        feature_index: int,
        threshold: float,
        x: int,
        y: int,
    ) -> None:
        """Draw manual split metrics."""
        self._draw_text("Manual split", x, y, self._font, TEXT_COLOR)
        y += 28

        rows = [
            ("Rule", f"x{feature_index + 1} <= {threshold:.3f}"),
        ]

        if manual_snapshot is None:
            rows.append(("Status", "invalid"))
        else:
            rows.extend(
                [
                    ("Status", "valid"),
                    ("Gain", f"{float(manual_snapshot.metrics['information_gain']):.4f}"),
                    ("Parent imp.", f"{float(manual_snapshot.metrics['parent_impurity']):.3f}"),
                    ("Left imp.", f"{float(manual_snapshot.metrics['left_impurity']):.3f}"),
                    ("Right imp.", f"{float(manual_snapshot.metrics['right_impurity']):.3f}"),
                ],
            )

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(value, x + 100, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

        if manual_error is not None:
            self._draw_text(
                manual_error[:32],
                x,
                y,
                self._small_font,
                ERROR_COLOR,
            )

    def _draw_root_split(self, snapshot: AlgorithmSnapshot, x: int, y: int) -> None:
        """Draw root split summary."""
        root = _extract_root(snapshot)

        self._draw_text("Root split", x, y, self._font, TEXT_COLOR)
        y += 28

        if root.is_leaf or root.split_evaluation is None:
            self._draw_text("No split selected.", x, y, self._small_font, MUTED_TEXT_COLOR)
            return

        candidate = root.split_evaluation.candidate
        rows = [
            ("Rule", f"x{candidate.feature_index + 1} <= {candidate.threshold:.3f}"),
            ("Gain", f"{root.split_evaluation.information_gain:.4f}"),
            ("Left imp.", f"{root.split_evaluation.left_impurity:.3f}"),
            ("Right imp.", f"{root.split_evaluation.right_impurity:.3f}"),
        ]

        for label, value in rows:
            self._draw_text(f"{label}:", x, y, self._small_font, MUTED_TEXT_COLOR)
            self._draw_text(value, x + 100, y, self._small_font, TEXT_COLOR)
            y += SMALL_TEXT_LINE_HEIGHT

    def _draw_bottom_panel(
        self,
        snapshot: AlgorithmSnapshot,
        *,
        mode: str,
        manual_error: str | None,
        challenge_result: DecisionTreeChallengeResult,
    ) -> None:
        """Draw keyboard controls and explanation."""
        x = BOTTOM_RECT.left + 24
        y = BOTTOM_RECT.top + 14

        controls = (
            "M: mode   D: dataset   G: criterion   F: manual feature   "
            "Q/E: threshold   Up/Down: depth   Left/Right: noise   S: seed"
        )
        self._draw_text(controls, x, y, self._small_font, TEXT_COLOR)

        explanation = _build_explanation(
            snapshot,
            mode=mode,
            manual_error=manual_error,
            challenge_result=challenge_result,
        )
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


def _extract_root(snapshot: AlgorithmSnapshot) -> DecisionTreeNode:
    """Extract and validate root node from a snapshot."""
    root = snapshot.visual_state["root"]

    if not isinstance(root, DecisionTreeNode):
        msg = "snapshot.visual_state['root'] must be a DecisionTreeNode."
        raise TypeError(msg)

    return root


def _build_explanation(
    snapshot: AlgorithmSnapshot,
    *,
    mode: str,
    manual_error: str | None,
    challenge_result: DecisionTreeChallengeResult,
) -> str:
    """Build short explanation for the current state."""
    if challenge_result.success and mode == MODE_AUTO_TREE:
        return challenge_result.message

    if mode == MODE_MANUAL_SPLIT:
        if manual_error is not None:
            return "Manual split is invalid because it does not create two useful children."

        return "Manual mode: move the split and compare impurity/gain with your intuition."

    accuracy = float(snapshot.metrics["training_accuracy"])
    max_depth = int(snapshot.metrics["max_depth"])
    leaf_count = int(snapshot.metrics["leaf_count"])

    return (
        f"Auto mode: recursive axis-aligned splits. "
        f"max_depth={max_depth}, leaves={leaf_count}, accuracy={accuracy:.2f}. "
        f"{challenge_result.status}: target accuracy={challenge_result.target_accuracy:.2f}."
    )


def _compute_world_bounds(features: FloatArray) -> WorldBounds:
    """Compute plot bounds with a small visual margin."""
    x_min = float(np.min(features[:, FEATURE_X_INDEX]))
    x_max = float(np.max(features[:, FEATURE_X_INDEX]))
    y_min = float(np.min(features[:, FEATURE_Y_INDEX]))
    y_max = float(np.max(features[:, FEATURE_Y_INDEX]))

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


def _split_bounds(
    bounds: WorldBounds,
    candidate: SplitCandidate,
) -> tuple[WorldBounds, WorldBounds]:
    """Split world bounds into left and right child regions."""
    threshold = candidate.threshold

    if candidate.feature_index == FEATURE_X_INDEX:
        left = WorldBounds(
            x_min=bounds.x_min,
            x_max=threshold,
            y_min=bounds.y_min,
            y_max=bounds.y_max,
        )
        right = WorldBounds(
            x_min=threshold,
            x_max=bounds.x_max,
            y_min=bounds.y_min,
            y_max=bounds.y_max,
        )
        return left, right

    left = WorldBounds(
        x_min=bounds.x_min,
        x_max=bounds.x_max,
        y_min=bounds.y_min,
        y_max=threshold,
    )
    right = WorldBounds(
        x_min=bounds.x_min,
        x_max=bounds.x_max,
        y_min=threshold,
        y_max=bounds.y_max,
    )

    return left, right


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


def _world_bounds_to_screen_rect(
    region_bounds: WorldBounds,
    global_bounds: WorldBounds,
    rect: pygame.Rect,
) -> pygame.Rect:
    """Convert world region bounds into a screen rectangle."""
    top_left = _world_to_screen(
        region_bounds.x_min,
        region_bounds.y_max,
        global_bounds,
        rect,
    )
    bottom_right = _world_to_screen(
        region_bounds.x_max,
        region_bounds.y_min,
        global_bounds,
        rect,
    )

    return pygame.Rect(
        top_left[0],
        top_left[1],
        max(1, bottom_right[0] - top_left[0]),
        max(1, bottom_right[1] - top_left[1]),
    )


def _majority_prediction(labels: IntArray) -> int:
    """Return majority class prediction for labels."""
    counts = np.bincount(labels)

    return int(np.argmax(counts))


def _manual_predictions(
    *,
    targets: IntArray,
    left_mask: BoolArray,
    right_mask: BoolArray,
    left_prediction: int,
    right_prediction: int,
) -> IntArray:
    """Build manual split predictions."""
    predictions = np.empty_like(targets)
    predictions[left_mask] = left_prediction
    predictions[right_mask] = right_prediction

    return predictions
