"""Pygame application loop for the Logistic Regression Boundary Lab demo."""

from __future__ import annotations

from typing import Final

import numpy as np
import pygame

from logistic_regression_boundary_lab.algorithm import (
    LogisticRegressionConfig,
    StepwiseLogisticRegression,
)
from logistic_regression_boundary_lab.dataset import (
    SyntheticBinaryClassificationConfig,
    make_synthetic_binary_classification_dataset,
)
from logistic_regression_boundary_lab.probability_grid import (
    DEFAULT_GRID_RESOLUTION,
    compute_probability_grid,
)
from logistic_regression_boundary_lab.renderer import (
    WINDOW_SIZE,
    LogisticRegressionRenderer,
)

FPS: Final[int] = 60
AUTO_STEP_INTERVAL_SECONDS: Final[float] = 0.08

DEFAULT_UI_LEARNING_RATE: Final[float] = 0.1
MIN_LEARNING_RATE: Final[float] = 0.01
MAX_LEARNING_RATE: Final[float] = 1.0
LEARNING_RATE_STEP: Final[float] = 0.02

DEFAULT_UI_THRESHOLD: Final[float] = 0.5
MIN_THRESHOLD: Final[float] = 0.05
MAX_THRESHOLD: Final[float] = 0.95
THRESHOLD_STEP: Final[float] = 0.05

DEFAULT_UI_MAX_STEPS: Final[int] = 200
DEFAULT_UI_SAMPLES_PER_CLASS: Final[int] = 70
DEFAULT_UI_NOISE_STD: Final[float] = 1.0
DEFAULT_UI_SEED: Final[int] = 42

NOISE_STEP: Final[float] = 0.2
MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 4.0
SEED_STEP: Final[int] = 1


class LogisticRegressionPygameApp:
    """Small Pygame application showing logistic regression training."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs — Logistic Regression Boundary Lab")

        self._clock = pygame.time.Clock()
        self._renderer = LogisticRegressionRenderer(self._screen)

        self._running = False
        self._should_quit = False
        self._time_since_last_step = 0.0

        self._learning_rate = DEFAULT_UI_LEARNING_RATE
        self._threshold = DEFAULT_UI_THRESHOLD
        self._noise_std = DEFAULT_UI_NOISE_STD
        self._seed = DEFAULT_UI_SEED

        self._model: StepwiseLogisticRegression
        self._dataset_config: SyntheticBinaryClassificationConfig

        self._reset_demo()

    def run(self) -> None:
        """Run the Pygame event loop."""
        while not self._should_quit:
            dt = self._clock.tick(FPS) / 1000.0

            self._handle_events()
            self._update(dt)
            self._draw()

        pygame.quit()

    def _reset_demo(self) -> None:
        """Reset dataset and model state using current parameters."""
        self._dataset_config = SyntheticBinaryClassificationConfig(
            samples_per_class=DEFAULT_UI_SAMPLES_PER_CLASS,
            noise_std=self._noise_std,
            seed=self._seed,
        )
        self._dataset = make_synthetic_binary_classification_dataset(self._dataset_config)

        self._model = StepwiseLogisticRegression(
            LogisticRegressionConfig(
                learning_rate=self._learning_rate,
                threshold=self._threshold,
                max_steps=DEFAULT_UI_MAX_STEPS,
            ),
        )
        self._model.reset(self._dataset)
        self._snapshot = self._model.snapshot()
        self._refresh_probability_grid()

        self._running = False
        self._time_since_last_step = 0.0

    def _refresh_probability_grid(self) -> None:
        """Recompute probability background for the current model state."""
        weights = np.asarray(self._snapshot.visual_state["weights"], dtype=float)
        bias = float(self._snapshot.visual_state["bias"])

        self._probability_grid = compute_probability_grid(
            self._dataset.features,
            weights=weights,
            bias=bias,
            resolution=DEFAULT_GRID_RESOLUTION,
        )

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
        elif event.key == pygame.K_SPACE:
            self._running = not self._running
        elif event.key == pygame.K_n:
            self._step_once()
        elif event.key == pygame.K_r:
            self._reset_demo()
        elif event.key == pygame.K_UP:
            self._increase_learning_rate()
        elif event.key == pygame.K_DOWN:
            self._decrease_learning_rate()
        elif event.key == pygame.K_e:
            self._increase_threshold()
        elif event.key == pygame.K_q:
            self._decrease_threshold()
        elif event.key == pygame.K_RIGHT:
            self._increase_noise()
        elif event.key == pygame.K_LEFT:
            self._decrease_noise()
        elif event.key == pygame.K_s:
            self._next_seed()

    def _increase_learning_rate(self) -> None:
        """Increase learning rate and reset the demo."""
        self._learning_rate = min(
            MAX_LEARNING_RATE,
            self._learning_rate + LEARNING_RATE_STEP,
        )
        self._reset_demo()

    def _decrease_learning_rate(self) -> None:
        """Decrease learning rate and reset the demo."""
        self._learning_rate = max(
            MIN_LEARNING_RATE,
            self._learning_rate - LEARNING_RATE_STEP,
        )
        self._reset_demo()

    def _increase_threshold(self) -> None:
        """Increase threshold and reset the demo."""
        self._threshold = min(
            MAX_THRESHOLD,
            self._threshold + THRESHOLD_STEP,
        )
        self._reset_demo()

    def _decrease_threshold(self) -> None:
        """Decrease threshold and reset the demo."""
        self._threshold = max(
            MIN_THRESHOLD,
            self._threshold - THRESHOLD_STEP,
        )
        self._reset_demo()

    def _increase_noise(self) -> None:
        """Increase dataset noise and reset the demo."""
        self._noise_std = min(
            MAX_NOISE_STD,
            self._noise_std + NOISE_STEP,
        )
        self._reset_demo()

    def _decrease_noise(self) -> None:
        """Decrease dataset noise and reset the demo."""
        self._noise_std = max(
            MIN_NOISE_STD,
            self._noise_std - NOISE_STEP,
        )
        self._reset_demo()

    def _next_seed(self) -> None:
        """Generate another dataset by changing the random seed."""
        self._seed += SEED_STEP
        self._reset_demo()

    def _update(self, dt: float) -> None:
        """Update automatic stepping state."""
        if not self._running or self._snapshot.done:
            return

        self._time_since_last_step += dt

        if self._time_since_last_step >= AUTO_STEP_INTERVAL_SECONDS:
            self._step_once()
            self._time_since_last_step = 0.0

    def _step_once(self) -> None:
        """Perform one training step."""
        if not self._snapshot.done:
            self._snapshot = self._model.step()
            self._refresh_probability_grid()

    def _draw(self) -> None:
        """Draw the current application state."""
        self._renderer.draw(
            self._snapshot,
            running=self._running,
            noise_std=self._noise_std,
            seed=self._seed,
            probability_grid=self._probability_grid,
        )


def main() -> None:
    """Run the Pygame visualization."""
    app = LogisticRegressionPygameApp()
    app.run()
