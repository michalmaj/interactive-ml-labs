"""Reusable Pygame scene for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from typing import Final

import pygame
from ml_lab_core import AlgorithmSnapshot

from random_forest_bagging_lab.baseline import (
    SingleTreeBaseline,
    SingleTreeBaselineConfig,
)
from random_forest_bagging_lab.challenge import (
    RandomForestChallenge,
    RandomForestChallengeResult,
)
from random_forest_bagging_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticTrainTestDatasetConfig,
    TrainTestDataset,
    make_synthetic_train_test_dataset,
)
from random_forest_bagging_lab.forest import RandomForestConfig, RandomForestModel
from random_forest_bagging_lab.renderer import WINDOW_SIZE, RandomForestRenderer
from random_forest_bagging_lab.report import (
    ModelComparisonReport,
    build_model_comparison_report,
)

DEFAULT_TRAIN_SAMPLES_PER_CLASS: Final[int] = 80
DEFAULT_TEST_SAMPLES_PER_CLASS: Final[int] = 80
DEFAULT_CLASS_DISTANCE: Final[float] = 4.0
DEFAULT_NOISE_STD: Final[float] = 0.65
DEFAULT_SEED: Final[int] = 42
DEFAULT_DATASET_KIND: Final[str] = DATASET_KIND_XOR
DEFAULT_TREE_COUNT: Final[int] = 25
DEFAULT_MAX_DEPTH: Final[int] = 2
DEFAULT_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 1.0
DEFAULT_CONFIDENCE_VIEW_ENABLED: Final[bool] = False

MIN_TREE_COUNT: Final[int] = 1
MAX_TREE_COUNT: Final[int] = 75
TREE_COUNT_STEP: Final[int] = 4

MIN_MAX_DEPTH: Final[int] = 1
MAX_MAX_DEPTH: Final[int] = 6
MAX_DEPTH_STEP: Final[int] = 1

MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 3.0
NOISE_STEP: Final[float] = 0.15

MIN_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 0.20
MAX_BOOTSTRAP_SAMPLE_RATIO: Final[float] = 1.00
BOOTSTRAP_SAMPLE_RATIO_STEP: Final[float] = 0.10

SEED_STEP: Final[int] = 1


class RandomForestScene:
    """Interactive scene comparing one tree with a random forest."""

    fixed_scene_size = WINDOW_SIZE

    def __init__(
        self,
        screen: pygame.Surface,
        *,
        present_frame: bool = True,
        language: str = "en",
    ) -> None:
        """Initialize scene state."""
        self._renderer = RandomForestRenderer(
            screen,
            present_frame=present_frame,
            language=language,
        )
        self._challenge = RandomForestChallenge()

        self._dataset_kind = DEFAULT_DATASET_KIND
        self._noise_std = DEFAULT_NOISE_STD
        self._seed = DEFAULT_SEED
        self._tree_count = DEFAULT_TREE_COUNT
        self._max_depth = DEFAULT_MAX_DEPTH
        self._bootstrap_sample_ratio = DEFAULT_BOOTSTRAP_SAMPLE_RATIO
        self._confidence_view_enabled = DEFAULT_CONFIDENCE_VIEW_ENABLED

        self._dataset: TrainTestDataset
        self._baseline_model: SingleTreeBaseline
        self._forest_model: RandomForestModel
        self._baseline_snapshot: AlgorithmSnapshot
        self._forest_snapshot: AlgorithmSnapshot
        self._comparison_report: ModelComparisonReport
        self._challenge_result: RandomForestChallengeResult

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
        """Advance scene state."""
        _ = dt

    def render(self) -> None:
        """Render the current scene frame."""
        self._renderer.draw(
            baseline_model=self._baseline_model,
            forest_model=self._forest_model,
            baseline_snapshot=self._baseline_snapshot,
            forest_snapshot=self._forest_snapshot,
            comparison_report=self._comparison_report,
            challenge_result=self._challenge_result,
            dataset_kind=self._dataset_kind,
            noise_std=self._noise_std,
            seed=self._seed,
            tree_count=self._tree_count,
            max_depth=self._max_depth,
            bootstrap_sample_ratio=self._bootstrap_sample_ratio,
            confidence_view_enabled=self._confidence_view_enabled,
        )

    def _reset_demo(self) -> None:
        """Regenerate dataset and refit both models."""
        self._dataset = make_synthetic_train_test_dataset(
            SyntheticTrainTestDatasetConfig(
                train_samples_per_class=DEFAULT_TRAIN_SAMPLES_PER_CLASS,
                test_samples_per_class=DEFAULT_TEST_SAMPLES_PER_CLASS,
                class_distance=DEFAULT_CLASS_DISTANCE,
                noise_std=self._noise_std,
                seed=self._seed,
                dataset_kind=self._dataset_kind,
            ),
        )
        self._fit_models()

    def _fit_models(self) -> None:
        """Fit baseline tree, random forest, and evaluate challenge status."""
        self._baseline_model = SingleTreeBaseline(
            SingleTreeBaselineConfig(max_depth=self._max_depth),
        )
        self._forest_model = RandomForestModel(
            RandomForestConfig(
                tree_count=self._tree_count,
                max_depth=self._max_depth,
                bootstrap_sample_ratio=self._bootstrap_sample_ratio,
                seed=self._seed,
            ),
        )

        self._baseline_snapshot = self._baseline_model.reset(self._dataset)
        self._forest_snapshot = self._forest_model.reset(self._dataset)
        self._comparison_report = build_model_comparison_report(
            single_tree_snapshot=self._baseline_snapshot,
            forest_snapshot=self._forest_snapshot,
        )
        self._challenge_result = self._challenge.evaluate(self._comparison_report)

    def _handle_keydown(self, event: pygame.event.Event) -> bool:
        """Dispatch keyboard actions."""
        if event.key == pygame.K_ESCAPE:
            return False

        key_actions = {
            pygame.K_r: self._reset_demo,
            pygame.K_d: self._toggle_dataset_kind,
            pygame.K_UP: self._increase_tree_count,
            pygame.K_DOWN: self._decrease_tree_count,
            pygame.K_w: self._increase_max_depth,
            pygame.K_s: self._decrease_max_depth,
            pygame.K_RIGHT: self._increase_noise,
            pygame.K_LEFT: self._decrease_noise,
            pygame.K_n: self._next_seed,
            pygame.K_b: self._increase_bootstrap_sample_ratio,
            pygame.K_v: self._decrease_bootstrap_sample_ratio,
            pygame.K_c: self._toggle_confidence_view,
        }

        action = key_actions.get(event.key)

        if action is not None:
            action()

        return True

    def _toggle_dataset_kind(self) -> None:
        """Toggle between axis-aligned and XOR datasets."""
        if self._dataset_kind == DATASET_KIND_AXIS_ALIGNED:
            self._dataset_kind = DATASET_KIND_XOR
        else:
            self._dataset_kind = DATASET_KIND_AXIS_ALIGNED

        self._reset_demo()

    def _increase_tree_count(self) -> None:
        """Increase forest tree count."""
        self._tree_count = min(MAX_TREE_COUNT, self._tree_count + TREE_COUNT_STEP)
        self._fit_models()

    def _decrease_tree_count(self) -> None:
        """Decrease forest tree count."""
        self._tree_count = max(MIN_TREE_COUNT, self._tree_count - TREE_COUNT_STEP)
        self._fit_models()

    def _increase_max_depth(self) -> None:
        """Increase tree max depth."""
        self._max_depth = min(MAX_MAX_DEPTH, self._max_depth + MAX_DEPTH_STEP)
        self._fit_models()

    def _decrease_max_depth(self) -> None:
        """Decrease tree max depth."""
        self._max_depth = max(MIN_MAX_DEPTH, self._max_depth - MAX_DEPTH_STEP)
        self._fit_models()

    def _increase_noise(self) -> None:
        """Increase dataset noise."""
        self._noise_std = min(MAX_NOISE_STD, self._noise_std + NOISE_STEP)
        self._reset_demo()

    def _decrease_noise(self) -> None:
        """Decrease dataset noise."""
        self._noise_std = max(MIN_NOISE_STD, self._noise_std - NOISE_STEP)
        self._reset_demo()

    def _increase_bootstrap_sample_ratio(self) -> None:
        """Increase bootstrap sample ratio used by the forest."""
        self._bootstrap_sample_ratio = min(
            MAX_BOOTSTRAP_SAMPLE_RATIO,
            self._bootstrap_sample_ratio + BOOTSTRAP_SAMPLE_RATIO_STEP,
        )
        self._bootstrap_sample_ratio = round(self._bootstrap_sample_ratio, 2)
        self._fit_models()

    def _decrease_bootstrap_sample_ratio(self) -> None:
        """Decrease bootstrap sample ratio used by the forest."""
        self._bootstrap_sample_ratio = max(
            MIN_BOOTSTRAP_SAMPLE_RATIO,
            self._bootstrap_sample_ratio - BOOTSTRAP_SAMPLE_RATIO_STEP,
        )
        self._bootstrap_sample_ratio = round(self._bootstrap_sample_ratio, 2)
        self._fit_models()

    def _toggle_confidence_view(self) -> None:
        """Toggle forest confidence-based region coloring."""
        self._confidence_view_enabled = not self._confidence_view_enabled

    def _next_seed(self) -> None:
        """Generate another dataset seed."""
        self._seed += SEED_STEP
        self._reset_demo()
