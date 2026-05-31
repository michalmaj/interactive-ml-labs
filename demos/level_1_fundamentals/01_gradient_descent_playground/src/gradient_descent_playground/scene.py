"""Reusable Pygame scene for the Gradient Descent Playground demo."""

from __future__ import annotations

from typing import Final

import pygame

from gradient_descent_playground.algorithm import (
    GradientDescentConfig,
    StepwiseLinearRegression,
)
from gradient_descent_playground.challenge import LossChallenge
from gradient_descent_playground.dataset import (
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)
from gradient_descent_playground.renderer import GradientDescentRenderer

AUTO_STEP_INTERVAL_SECONDS: Final[float] = 0.08

DEFAULT_UI_LEARNING_RATE: Final[float] = 0.03
DEFAULT_UI_MAX_STEPS: Final[int] = 160
DEFAULT_UI_SAMPLE_COUNT: Final[int] = 80
DEFAULT_UI_NOISE_STD: Final[float] = 0.8
DEFAULT_UI_SEED: Final[int] = 42

LEARNING_RATE_STEP: Final[float] = 0.01
MIN_LEARNING_RATE: Final[float] = 0.001
MAX_LEARNING_RATE: Final[float] = 0.25

NOISE_STEP: Final[float] = 0.2
MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 5.0

SEED_STEP: Final[int] = 1


class GradientDescentScene:
    """Interactive scene for the Gradient Descent Playground demo."""

    def __init__(self, screen: pygame.Surface, *, present_frame: bool = True) -> None:
        """Initialize the scene with a target screen."""
        self._renderer = GradientDescentRenderer(screen, present_frame=present_frame)
        self._challenge = LossChallenge()

        self._running = False
        self._time_since_last_step = 0.0

        self._learning_rate = DEFAULT_UI_LEARNING_RATE
        self._noise_std = DEFAULT_UI_NOISE_STD
        self._seed = DEFAULT_UI_SEED

        self._model: StepwiseLinearRegression
        self._dataset_config: SyntheticRegressionConfig

        self._rebuild_model()
        self._reset_demo()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle one Pygame event.

        Returns:
            True when the standalone app should keep running, false when the
            scene requests closing the standalone app.
        """
        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event)

        return True

    def update(self, dt: float) -> None:
        """Update automatic stepping state."""
        if not self._running or self._snapshot.done or self._challenge_result.success:
            return

        self._time_since_last_step += dt

        if self._time_since_last_step >= AUTO_STEP_INTERVAL_SECONDS:
            self._step_once()
            self._time_since_last_step = 0.0

    def render(self) -> None:
        """Render the current scene frame."""
        self._renderer.draw(
            self._snapshot,
            running=self._running,
            noise_std=self._noise_std,
            seed=self._seed,
            challenge_result=self._challenge_result,
        )

    def _rebuild_model(self) -> None:
        """Recreate the model using current training parameters."""
        self._model = StepwiseLinearRegression(
            GradientDescentConfig(
                learning_rate=self._learning_rate,
                max_steps=DEFAULT_UI_MAX_STEPS,
            ),
        )

    def _rebuild_dataset_config(self) -> None:
        """Recreate dataset configuration using current data parameters."""
        self._dataset_config = SyntheticRegressionConfig(
            sample_count=DEFAULT_UI_SAMPLE_COUNT,
            noise_std=self._noise_std,
            seed=self._seed,
        )

    def _reset_demo(self) -> None:
        """Reset dataset, model, and challenge state using current parameters."""
        self._rebuild_model()
        self._rebuild_dataset_config()

        self._dataset = make_synthetic_regression_dataset(self._dataset_config)
        self._model.reset(self._dataset)
        self._snapshot = self._model.snapshot()
        self._challenge_result = self._challenge.evaluate(self._snapshot)

        self._running = False
        self._time_since_last_step = 0.0

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """Handle keyboard shortcuts."""
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_SPACE:
            self._running = not self._running
        elif event.key == pygame.K_n:
            self._step_once()
        elif event.key == pygame.K_r:
            self._reset_demo()
        elif event.key == pygame.K_UP:
            self._increase_learning_rate()
        elif event.key == pygame.K_DOWN:
            self._decrease_learning_rate()
        elif event.key == pygame.K_RIGHT:
            self._increase_noise()
        elif event.key == pygame.K_LEFT:
            self._decrease_noise()
        elif event.key == pygame.K_s:
            self._next_seed()

        return True

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

    def _increase_noise(self) -> None:
        """Increase data noise and reset the demo."""
        self._noise_std = min(
            MAX_NOISE_STD,
            self._noise_std + NOISE_STEP,
        )
        self._reset_demo()

    def _decrease_noise(self) -> None:
        """Decrease data noise and reset the demo."""
        self._noise_std = max(
            MIN_NOISE_STD,
            self._noise_std - NOISE_STEP,
        )
        self._reset_demo()

    def _next_seed(self) -> None:
        """Generate another dataset by changing the random seed."""
        self._seed += SEED_STEP
        self._reset_demo()

    def _step_once(self) -> None:
        """Perform one algorithm step and update the challenge state."""
        if not self._snapshot.done and not self._challenge_result.success:
            self._snapshot = self._model.step()
            self._challenge_result = self._challenge.evaluate(self._snapshot)
