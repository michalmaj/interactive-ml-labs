"""Reusable Pygame scene for the k-NN Vote Map demo."""

from __future__ import annotations

from typing import Final

import numpy as np
import pygame

from knn_vote_map.challenge import KNNAccuracyChallenge, KNNAccuracyChallengeResult
from knn_vote_map.classifier import KNearestNeighborsClassifier, KNearestNeighborsConfig
from knn_vote_map.dataset import (
    SyntheticClassificationConfig,
    make_synthetic_classification_dataset,
)
from knn_vote_map.decision_grid import DEFAULT_GRID_RESOLUTION, compute_decision_grid
from knn_vote_map.renderer import KNNVoteMapRenderer

DEFAULT_UI_K: Final[int] = 5
MIN_K: Final[int] = 1
MAX_K: Final[int] = 21
K_STEP: Final[int] = 2

DEFAULT_UI_SAMPLES_PER_CLASS: Final[int] = 60
DEFAULT_UI_TEST_SAMPLES_PER_CLASS: Final[int] = 40
DEFAULT_UI_NOISE_STD: Final[float] = 0.9
DEFAULT_UI_SEED: Final[int] = 42

NOISE_STEP: Final[float] = 0.2
MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 4.0
SEED_STEP: Final[int] = 1
TEST_SEED_OFFSET: Final[int] = 50_000

QUERY_MIN: Final[float] = -5.5
QUERY_MAX: Final[float] = 5.5
LEFT_MOUSE_BUTTON: Final[int] = 1


class KNNVoteMapScene:
    """Interactive scene for the k-NN Vote Map demo."""

    def __init__(
        self,
        screen: pygame.Surface,
        *,
        present_frame: bool = True,
        language: str = "en",
    ) -> None:
        """Initialize the scene with a target screen."""
        self._renderer = KNNVoteMapRenderer(
            screen,
            present_frame=present_frame,
            language=language,
        )
        self._challenge = KNNAccuracyChallenge()

        self._k = DEFAULT_UI_K
        self._noise_std = DEFAULT_UI_NOISE_STD
        self._seed = DEFAULT_UI_SEED

        self._classifier: KNearestNeighborsClassifier
        self._dataset_config: SyntheticClassificationConfig
        self._challenge_result: KNNAccuracyChallengeResult
        self._rng = np.random.default_rng(self._seed)

        self._reset_demo()

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle one Pygame event.

        Returns:
            True when the standalone app should keep running, false when the
            scene requests closing the standalone app.
        """
        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event)
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_button_down(event)

        return True

    def update(self, dt: float) -> None:
        """Advance scene state."""
        _ = dt

    def render(self) -> None:
        """Render the current scene frame."""
        self._renderer.draw(
            self._snapshot,
            noise_std=self._noise_std,
            seed=self._seed,
            decision_grid=self._decision_grid,
            challenge_result=self._challenge_result,
        )

    def _reset_demo(self) -> None:
        """Reset dataset, classifier, decision grid, and challenge state."""
        self._dataset_config = SyntheticClassificationConfig(
            samples_per_class=DEFAULT_UI_SAMPLES_PER_CLASS,
            noise_std=self._noise_std,
            seed=self._seed,
        )
        self._dataset = make_synthetic_classification_dataset(self._dataset_config)

        self._classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=self._k))
        self._classifier.fit(self._dataset)
        self._snapshot = self._classifier.snapshot()
        self._decision_grid = compute_decision_grid(
            self._dataset.features,
            self._dataset.targets,
            k=self._k,
            resolution=DEFAULT_GRID_RESOLUTION,
        )
        self._challenge_result = self._evaluate_challenge()
        self._rng = np.random.default_rng(self._seed + 10_000)

    def _evaluate_challenge(self) -> KNNAccuracyChallengeResult:
        """Evaluate current k-NN configuration on a hidden synthetic test set."""
        test_config = SyntheticClassificationConfig(
            samples_per_class=DEFAULT_UI_TEST_SAMPLES_PER_CLASS,
            noise_std=self._noise_std,
            seed=self._seed + TEST_SEED_OFFSET,
        )
        test_dataset = make_synthetic_classification_dataset(test_config)

        evaluator = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=self._k))
        evaluator.fit(self._dataset)

        predictions = evaluator.predict(test_dataset.features)

        return self._challenge.evaluate(
            y_true=test_dataset.targets,
            y_pred=predictions,
        )

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """Handle keyboard shortcuts."""
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_n:
            self._classify_random_query_point()
        elif event.key == pygame.K_r:
            self._reset_demo()
        elif event.key == pygame.K_UP:
            self._increase_k()
        elif event.key == pygame.K_DOWN:
            self._decrease_k()
        elif event.key == pygame.K_RIGHT:
            self._increase_noise()
        elif event.key == pygame.K_LEFT:
            self._decrease_noise()
        elif event.key == pygame.K_s:
            self._next_seed()

        return True

    def _handle_mouse_button_down(self, event: pygame.event.Event) -> None:
        """Handle mouse clicks on the vote map."""
        if event.button != LEFT_MOUSE_BUTTON:
            return

        world_position = self._renderer.screen_to_world(event.pos)

        if world_position is None:
            return

        self._classify_query_point(world_position)

    def _classify_random_query_point(self) -> None:
        """Sample and classify one random query point."""
        query_point = self._rng.uniform(
            low=QUERY_MIN,
            high=QUERY_MAX,
            size=2,
        )
        self._classify_query_point(query_point)

    def _classify_query_point(self, query_point: tuple[float, float] | np.ndarray) -> None:
        """Classify one query point and update the current snapshot."""
        self._classifier.predict_one(query_point)
        self._snapshot = self._classifier.snapshot()

    def _increase_k(self) -> None:
        """Increase k and reset classifier state."""
        self._k = min(MAX_K, self._k + K_STEP)
        self._reset_demo()

    def _decrease_k(self) -> None:
        """Decrease k and reset classifier state."""
        self._k = max(MIN_K, self._k - K_STEP)
        self._reset_demo()

    def _increase_noise(self) -> None:
        """Increase dataset noise and reset the demo."""
        self._noise_std = min(MAX_NOISE_STD, self._noise_std + NOISE_STEP)
        self._reset_demo()

    def _decrease_noise(self) -> None:
        """Decrease dataset noise and reset the demo."""
        self._noise_std = max(MIN_NOISE_STD, self._noise_std - NOISE_STEP)
        self._reset_demo()

    def _next_seed(self) -> None:
        """Generate another dataset by changing the random seed."""
        self._seed += SEED_STEP
        self._reset_demo()
