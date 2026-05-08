"""Pygame application loop for the Decision Tree Splitter demo."""

from __future__ import annotations

from typing import Final

import pygame

from decision_tree_splitter.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticDecisionTreeDatasetConfig,
    make_synthetic_decision_tree_dataset,
)
from decision_tree_splitter.renderer import WINDOW_SIZE, DecisionTreeRenderer
from decision_tree_splitter.split import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    ImpurityCriterion,
)
from decision_tree_splitter.tree import DecisionTreeConfig, RecursiveDecisionTree

FPS: Final[int] = 60

DEFAULT_UI_SAMPLES_PER_CLASS: Final[int] = 80
DEFAULT_UI_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_UI_NOISE_STD: Final[float] = 0.45
DEFAULT_UI_SEED: Final[int] = 42
DEFAULT_UI_MAX_DEPTH: Final[int] = 2
DEFAULT_UI_DATASET_KIND: Final[str] = DATASET_KIND_AXIS_ALIGNED
DEFAULT_UI_CRITERION: Final[ImpurityCriterion] = CRITERION_GINI

MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 3.0
NOISE_STEP: Final[float] = 0.15

MIN_MAX_DEPTH: Final[int] = 1
MAX_MAX_DEPTH: Final[int] = 6
MAX_DEPTH_STEP: Final[int] = 1
SEED_STEP: Final[int] = 1


class DecisionTreePygameApp:
    """Small Pygame application showing decision tree splits."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs — Decision Tree Splitter")

        self._clock = pygame.time.Clock()
        self._renderer = DecisionTreeRenderer(self._screen)

        self._should_quit = False
        self._dataset_kind = DEFAULT_UI_DATASET_KIND
        self._criterion = DEFAULT_UI_CRITERION
        self._noise_std = DEFAULT_UI_NOISE_STD
        self._seed = DEFAULT_UI_SEED
        self._max_depth = DEFAULT_UI_MAX_DEPTH

        self._reset_demo()

    def run(self) -> None:
        """Run the Pygame event loop."""
        while not self._should_quit:
            self._clock.tick(FPS)

            self._handle_events()
            self._draw()

        pygame.quit()

    def _reset_demo(self) -> None:
        """Reset dataset and tree using current UI parameters."""
        dataset_config = SyntheticDecisionTreeDatasetConfig(
            samples_per_class=DEFAULT_UI_SAMPLES_PER_CLASS,
            class_distance=DEFAULT_UI_CLASS_DISTANCE,
            noise_std=self._noise_std,
            seed=self._seed,
            dataset_kind=self._dataset_kind,
        )
        self._dataset = make_synthetic_decision_tree_dataset(dataset_config)

        tree = RecursiveDecisionTree(
            DecisionTreeConfig(
                criterion=self._criterion,
                max_depth=self._max_depth,
            ),
        )
        self._snapshot = tree.reset(self._dataset)

    def _handle_events(self) -> None:
        """Handle Pygame input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """Handle keyboard shortcuts."""
        if event.key == pygame.K_ESCAPE:
            self._should_quit = True
        elif event.key == pygame.K_r:
            self._reset_demo()
        elif event.key == pygame.K_d:
            self._toggle_dataset_kind()
        elif event.key == pygame.K_g:
            self._toggle_criterion()
        elif event.key == pygame.K_UP:
            self._increase_max_depth()
        elif event.key == pygame.K_DOWN:
            self._decrease_max_depth()
        elif event.key == pygame.K_RIGHT:
            self._increase_noise()
        elif event.key == pygame.K_LEFT:
            self._decrease_noise()
        elif event.key == pygame.K_s:
            self._next_seed()

    def _toggle_dataset_kind(self) -> None:
        """Toggle between axis-aligned and XOR datasets."""
        if self._dataset_kind == DATASET_KIND_AXIS_ALIGNED:
            self._dataset_kind = DATASET_KIND_XOR
        else:
            self._dataset_kind = DATASET_KIND_AXIS_ALIGNED

        self._reset_demo()

    def _toggle_criterion(self) -> None:
        """Toggle between Gini and entropy split scoring."""
        if self._criterion == CRITERION_GINI:
            self._criterion = CRITERION_ENTROPY
        else:
            self._criterion = CRITERION_GINI

        self._reset_demo()

    def _increase_max_depth(self) -> None:
        """Increase maximum tree depth."""
        self._max_depth = min(MAX_MAX_DEPTH, self._max_depth + MAX_DEPTH_STEP)
        self._reset_demo()

    def _decrease_max_depth(self) -> None:
        """Decrease maximum tree depth."""
        self._max_depth = max(MIN_MAX_DEPTH, self._max_depth - MAX_DEPTH_STEP)
        self._reset_demo()

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
            self._snapshot,
            dataset_kind=self._dataset_kind,
            noise_std=self._noise_std,
            seed=self._seed,
        )


def main() -> None:
    """Run the Pygame visualization."""
    app = DecisionTreePygameApp()
    app.run()
