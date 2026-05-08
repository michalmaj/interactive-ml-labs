"""Pygame application loop for the Decision Tree Splitter demo."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pygame
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

from decision_tree_splitter.challenge import (
    DecisionTreeChallenge,
    DecisionTreeChallengeResult,
)
from decision_tree_splitter.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from decision_tree_splitter.manual_split import ManualSplitConfig, ManualSplitPrototype
from decision_tree_splitter.renderer import WINDOW_SIZE, DecisionTreeRenderer
from decision_tree_splitter.split import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    ImpurityCriterion,
)
from decision_tree_splitter.tree import DecisionTreeConfig, RecursiveDecisionTree

FPS: Final[int] = 60

MODE_AUTO_TREE: Final[str] = "auto_tree"
MODE_MANUAL_SPLIT: Final[str] = "manual_split"

DEFAULT_UI_SAMPLES_PER_CLASS: Final[int] = 80
DEFAULT_UI_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_UI_NOISE_STD: Final[float] = 0.45
DEFAULT_UI_SEED: Final[int] = 42
DEFAULT_UI_MAX_DEPTH: Final[int] = 2
DEFAULT_UI_DATASET_KIND: Final[str] = DATASET_KIND_AXIS_ALIGNED
DEFAULT_UI_CRITERION: Final[ImpurityCriterion] = CRITERION_GINI
DEFAULT_MANUAL_FEATURE_INDEX: Final[int] = 0
DEFAULT_MANUAL_THRESHOLD: Final[float] = 0.0

MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 3.0
NOISE_STEP: Final[float] = 0.15

MIN_MAX_DEPTH: Final[int] = 1
MAX_MAX_DEPTH: Final[int] = 6
MAX_DEPTH_STEP: Final[int] = 1
SEED_STEP: Final[int] = 1

FEATURE_X_INDEX: Final[int] = 0
FEATURE_Y_INDEX: Final[int] = 1
MANUAL_THRESHOLD_STEP: Final[float] = 0.15


class DecisionTreePygameApp:
    """Small Pygame application showing decision tree splits."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs — Decision Tree Splitter")

        self._clock = pygame.time.Clock()
        self._renderer = DecisionTreeRenderer(self._screen)
        self._challenge = DecisionTreeChallenge()

        self._should_quit = False
        self._mode = MODE_AUTO_TREE
        self._dataset_kind = DEFAULT_UI_DATASET_KIND
        self._criterion = DEFAULT_UI_CRITERION
        self._noise_std = DEFAULT_UI_NOISE_STD
        self._seed = DEFAULT_UI_SEED
        self._max_depth = DEFAULT_UI_MAX_DEPTH
        self._manual_feature_index = DEFAULT_MANUAL_FEATURE_INDEX
        self._manual_threshold = DEFAULT_MANUAL_THRESHOLD

        self._dataset: DatasetSnapshot
        self._tree_snapshot: AlgorithmSnapshot
        self._manual_snapshot: AlgorithmSnapshot | None
        self._manual_error: str | None
        self._challenge_result: DecisionTreeChallengeResult

        self._reset_demo()

    def run(self) -> None:
        """Run the Pygame event loop."""
        while not self._should_quit:
            self._clock.tick(FPS)

            self._handle_events()
            self._draw()

        pygame.quit()

    def _reset_demo(self) -> None:
        """Reset dataset, tree, manual split state, and challenge state."""
        dataset_config = SyntheticDecisionTreeDatasetConfig(
            samples_per_class=DEFAULT_UI_SAMPLES_PER_CLASS,
            class_distance=DEFAULT_UI_CLASS_DISTANCE,
            noise_std=self._noise_std,
            seed=self._seed,
            dataset_kind=self._dataset_kind,
        )
        self._dataset = make_synthetic_decision_tree_dataset(dataset_config)
        self._refresh_tree_snapshot()
        self._refresh_manual_snapshot()

    def _refresh_tree_snapshot(self) -> None:
        """Fit the recursive tree using current UI parameters."""
        tree = RecursiveDecisionTree(
            DecisionTreeConfig(
                criterion=self._criterion,
                max_depth=self._max_depth,
            ),
        )
        self._tree_snapshot = tree.reset(self._dataset)
        self._refresh_challenge_result()

    def _refresh_challenge_result(self) -> None:
        """Evaluate current tree against the active challenge."""
        self._challenge_result = self._challenge.evaluate(
            snapshot=self._tree_snapshot,
            dataset_kind=self._dataset_kind,
        )

    def _refresh_manual_snapshot(self) -> None:
        """Evaluate the current manual split."""
        manual = ManualSplitPrototype(
            ManualSplitConfig(
                feature_index=self._manual_feature_index,
                threshold=self._manual_threshold,
                criterion=self._criterion,
            ),
        )

        try:
            self._manual_snapshot = manual.reset(self._dataset)
            self._manual_error = None
        except ValueError as error:
            self._manual_snapshot = None
            self._manual_error = str(error)

    def _handle_events(self) -> None:
        """Handle Pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """Handle keyboard shortcuts."""
        key_actions: dict[int, Callable[[], None]] = {
            pygame.K_ESCAPE: self._quit,
            pygame.K_r: self._reset_demo,
            pygame.K_m: self._toggle_mode,
            pygame.K_d: self._toggle_dataset_kind,
            pygame.K_g: self._toggle_criterion,
            pygame.K_f: self._toggle_manual_feature,
            pygame.K_q: self._decrease_manual_threshold,
            pygame.K_e: self._increase_manual_threshold,
            pygame.K_UP: self._increase_max_depth,
            pygame.K_DOWN: self._decrease_max_depth,
            pygame.K_RIGHT: self._increase_noise,
            pygame.K_LEFT: self._decrease_noise,
            pygame.K_s: self._next_seed,
        }

        action = key_actions.get(event.key)

        if action is not None:
            action()

    def _quit(self) -> None:
        """Request application shutdown."""
        self._should_quit = True

    def _toggle_mode(self) -> None:
        """Toggle between automatic tree mode and manual split mode."""
        if self._mode == MODE_AUTO_TREE:
            self._mode = MODE_MANUAL_SPLIT
        else:
            self._mode = MODE_AUTO_TREE

    def _toggle_dataset_kind(self) -> None:
        """Toggle between axis-aligned and XOR datasets."""
        if self._dataset_kind == DATASET_KIND_AXIS_ALIGNED:
            self._dataset_kind = DATASET_KIND_XOR
        else:
            self._dataset_kind = DATASET_KIND_AXIS_ALIGNED

        self._manual_threshold = DEFAULT_MANUAL_THRESHOLD
        self._reset_demo()

    def _toggle_criterion(self) -> None:
        """Toggle between Gini and entropy split scoring."""
        if self._criterion == CRITERION_GINI:
            self._criterion = CRITERION_ENTROPY
        else:
            self._criterion = CRITERION_GINI

        self._refresh_tree_snapshot()
        self._refresh_manual_snapshot()

    def _toggle_manual_feature(self) -> None:
        """Toggle manual split feature between x1 and x2."""
        if self._manual_feature_index == FEATURE_X_INDEX:
            self._manual_feature_index = FEATURE_Y_INDEX
        else:
            self._manual_feature_index = FEATURE_X_INDEX

        self._manual_threshold = DEFAULT_MANUAL_THRESHOLD
        self._refresh_manual_snapshot()

    def _increase_manual_threshold(self) -> None:
        """Move manual split threshold upward."""
        self._manual_threshold += MANUAL_THRESHOLD_STEP
        self._refresh_manual_snapshot()

    def _decrease_manual_threshold(self) -> None:
        """Move manual split threshold downward."""
        self._manual_threshold -= MANUAL_THRESHOLD_STEP
        self._refresh_manual_snapshot()

    def _increase_max_depth(self) -> None:
        """Increase maximum tree depth."""
        self._max_depth = min(MAX_MAX_DEPTH, self._max_depth + MAX_DEPTH_STEP)
        self._refresh_tree_snapshot()

    def _decrease_max_depth(self) -> None:
        """Decrease maximum tree depth."""
        self._max_depth = max(MIN_MAX_DEPTH, self._max_depth - MAX_DEPTH_STEP)
        self._refresh_tree_snapshot()

    def _increase_noise(self) -> None:
        """Increase dataset noise."""
        self._noise_std = min(MAX_NOISE_STD, self._noise_std + NOISE_STEP)
        self._reset_demo()

    def _decrease_noise(self) -> None:
        """Decrease dataset noise."""
        self._noise_std = max(MIN_NOISE_STD, self._noise_std - NOISE_STEP)
        self._reset_demo()

    def _next_seed(self) -> None:
        """Generate another dataset by changing the random seed."""
        self._seed += SEED_STEP
        self._reset_demo()

    def _draw(self) -> None:
        """Draw current application state."""
        self._renderer.draw(
            self._tree_snapshot,
            dataset_kind=self._dataset_kind,
            noise_std=self._noise_std,
            seed=self._seed,
            mode=self._mode,
            manual_snapshot=self._manual_snapshot,
            manual_error=self._manual_error,
            manual_feature_index=self._manual_feature_index,
            manual_threshold=self._manual_threshold,
            challenge_result=self._challenge_result,
        )


def main() -> None:
    """Run the Pygame visualization."""
    app = DecisionTreePygameApp()
    app.run()
