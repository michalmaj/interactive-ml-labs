"""Pygame application loop for the Gradient Descent Playground demo."""

from __future__ import annotations

from typing import Final

import pygame

from gradient_descent_playground.algorithm import (
    GradientDescentConfig,
    StepwiseLinearRegression,
)
from gradient_descent_playground.dataset import (
    SyntheticRegressionConfig,
    make_synthetic_regression_dataset,
)
from gradient_descent_playground.renderer import (
    WINDOW_SIZE,
    GradientDescentRenderer,
)

FPS: Final[int] = 60
AUTO_STEP_INTERVAL_SECONDS: Final[float] = 0.08

DEFAULT_UI_LEARNING_RATE: Final[float] = 0.03
DEFAULT_UI_MAX_STEPS: Final[int] = 160
DEFAULT_UI_SAMPLE_COUNT: Final[int] = 80
DEFAULT_UI_NOISE_STD: Final[float] = 0.8
DEFAULT_UI_SEED: Final[int] = 42


class GradientDescentPygameApp:
    """Small Pygame application showing stepwise gradient descent."""

    def __init__(self) -> None:
        """Initialize the application state."""
        pygame.init()

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Interactive ML Labs — Gradient Descent Playground")

        self._clock = pygame.time.Clock()
        self._renderer = GradientDescentRenderer(self._screen)

        self._running = False
        self._should_quit = False
        self._time_since_last_step = 0.0

        self._model = StepwiseLinearRegression(
            GradientDescentConfig(
                learning_rate=DEFAULT_UI_LEARNING_RATE,
                max_steps=DEFAULT_UI_MAX_STEPS,
            ),
        )
        self._dataset_config = SyntheticRegressionConfig(
            sample_count=DEFAULT_UI_SAMPLE_COUNT,
            noise_std=DEFAULT_UI_NOISE_STD,
            seed=DEFAULT_UI_SEED,
        )

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
        """Reset dataset and model state."""
        self._dataset = make_synthetic_regression_dataset(self._dataset_config)
        self._model.reset(self._dataset)
        self._snapshot = self._model.snapshot()
        self._running = False
        self._time_since_last_step = 0.0

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

    def _update(self, dt: float) -> None:
        """Update automatic stepping state."""
        if not self._running or self._snapshot.done:
            return

        self._time_since_last_step += dt

        if self._time_since_last_step >= AUTO_STEP_INTERVAL_SECONDS:
            self._step_once()
            self._time_since_last_step = 0.0

    def _step_once(self) -> None:
        """Perform one algorithm step."""
        if not self._snapshot.done:
            self._snapshot = self._model.step()

    def _draw(self) -> None:
        """Draw the current application state."""
        self._renderer.draw(self._snapshot, running=self._running)


def main() -> None:
    """Run the Pygame visualization."""
    app = GradientDescentPygameApp()
    app.run()
