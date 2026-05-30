"""Pygame application for the Boosting Mistake Lab demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pygame

from boosting_mistake_lab.dataset import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    SyntheticWeightedDatasetConfig,
    WeightedTrainTestDataset,
    make_synthetic_weighted_dataset,
)
from boosting_mistake_lab.export import (
    DecisionBoundaryExportConfig,
    write_decision_boundary_export,
)
from boosting_mistake_lab.presets import (
    CUSTOM_PRESET_NAME,
    DEFAULT_PRESET_NUMBER,
    get_next_preset_number,
    get_preset_by_number,
)
from boosting_mistake_lab.renderer import (
    WINDOW_SIZE,
    BoostingRenderer,
    BoostingRenderState,
)
from boosting_mistake_lab.trainer import (
    BoostingTrainer,
    BoostingTrainerConfig,
    BoostingTrainerResult,
)

FPS: Final[int] = 30

DEFAULT_PRESET = get_preset_by_number(DEFAULT_PRESET_NUMBER)

MIN_ROUND_COUNT: Final[int] = 1
MAX_ROUND_COUNT: Final[int] = 12

MIN_MIN_SAMPLES_LEAF: Final[int] = 1
MAX_MIN_SAMPLES_LEAF: Final[int] = 8

MIN_NOISE_STD: Final[float] = 0.0
MAX_NOISE_STD: Final[float] = 1.80
NOISE_STEP: Final[float] = 0.10

EXPORT_DIR: Final[Path] = Path("exports")
EXPORT_FILENAME: Final[str] = "boosting_mistake_lab_decision_boundary.json"
EXPORT_GRID_RESOLUTION: Final[int] = 50


@dataclass(slots=True)
class BoostingPygameState:
    """Mutable Pygame demo state."""

    dataset_kind: str = DEFAULT_PRESET.dataset_kind
    round_count: int = DEFAULT_PRESET.round_count
    selected_stage: int = DEFAULT_PRESET.selected_stage
    min_samples_leaf: int = DEFAULT_PRESET.min_samples_leaf
    noise_std: float = DEFAULT_PRESET.noise_std
    seed: int = DEFAULT_PRESET.seed
    confidence_view_enabled: bool = DEFAULT_PRESET.confidence_view_enabled
    preset_number: int = DEFAULT_PRESET_NUMBER
    preset_name: str = DEFAULT_PRESET.name


class BoostingPygameApp:
    """Interactive Pygame app for the Boosting Mistake Lab demo."""

    def __init__(self) -> None:
        """Initialize the Pygame app."""
        pygame.init()
        pygame.display.set_caption("Boosting Mistake Lab")

        self._screen = pygame.display.set_mode(WINDOW_SIZE)
        self._clock = pygame.time.Clock()
        self._renderer = BoostingRenderer(self._screen)
        self._running = True
        self._state = BoostingPygameState()
        self._dataset: WeightedTrainTestDataset
        self._trainer_result: BoostingTrainerResult

        self._clamp_selected_stage()
        self._rebuild_demo()

    def run(self) -> None:
        """Run the main event loop."""
        try:
            while self._running:
                self._handle_events()
                self._renderer.draw(self._render_state())
                self._clock.tick(FPS)
        finally:
            pygame.quit()

    def _export_current_state(self) -> None:
        """Export current selected-stage decision boundary to JSON."""
        result = write_decision_boundary_export(
            dataset=self._dataset,
            trainer_result=self._trainer_result,
            config=DecisionBoundaryExportConfig(
                selected_stage=self._state.selected_stage,
                grid_resolution=EXPORT_GRID_RESOLUTION,
            ),
            output_path=EXPORT_DIR / EXPORT_FILENAME,
        )

        print(f"Exported Boosting Mistake Lab state to {result.output_path}")

    def _handle_events(self) -> None:
        """Handle all pending Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """Handle keyboard shortcuts."""
        handlers: dict[int, Callable[[], None]] = {
            pygame.K_ESCAPE: self._quit,
            pygame.K_1: lambda: self._apply_preset_by_number(1),
            pygame.K_2: lambda: self._apply_preset_by_number(2),
            pygame.K_3: lambda: self._apply_preset_by_number(3),
            pygame.K_4: lambda: self._apply_preset_by_number(4),
            pygame.K_p: self._apply_next_preset,
            pygame.K_d: self._toggle_dataset,
            pygame.K_UP: self._increase_selected_stage,
            pygame.K_DOWN: self._decrease_selected_stage,
            pygame.K_PAGEUP: self._increase_round_count,
            pygame.K_PAGEDOWN: self._decrease_round_count,
            pygame.K_EQUALS: self._increase_round_count,
            pygame.K_MINUS: self._decrease_round_count,
            pygame.K_w: self._increase_min_samples_leaf,
            pygame.K_s: self._decrease_min_samples_leaf,
            pygame.K_RIGHT: self._increase_noise,
            pygame.K_LEFT: self._decrease_noise,
            pygame.K_n: self._next_seed,
            pygame.K_c: self._toggle_confidence_view,
            pygame.K_r: self._reset_defaults,
            pygame.K_e: self._export_current_state,
        }
        handler = handlers.get(event.key)

        if handler is not None:
            handler()

    def _render_state(self) -> BoostingRenderState:
        """Create immutable render state."""
        return BoostingRenderState(
            dataset=self._dataset,
            trainer_result=self._trainer_result,
            dataset_kind=self._state.dataset_kind,
            noise_std=self._state.noise_std,
            seed=self._state.seed,
            round_count=self._state.round_count,
            selected_stage=self._state.selected_stage,
            min_samples_leaf=self._state.min_samples_leaf,
            confidence_view_enabled=self._state.confidence_view_enabled,
            preset_name=self._state.preset_name,
        )

    def _rebuild_demo(self) -> None:
        """Regenerate dataset and refit boosting trainer."""
        self._dataset = make_synthetic_weighted_dataset(
            SyntheticWeightedDatasetConfig(
                noise_std=self._state.noise_std,
                seed=self._state.seed,
                dataset_kind=self._state.dataset_kind,
            ),
        )
        trainer = BoostingTrainer(
            BoostingTrainerConfig(
                round_count=self._state.round_count,
                min_samples_leaf=self._state.min_samples_leaf,
            ),
        )
        self._trainer_result = trainer.reset(self._dataset)
        self._clamp_selected_stage()

    def _quit(self) -> None:
        """Stop the app."""
        self._running = False

    def _apply_preset_by_number(self, preset_number: int) -> None:
        """Apply one preset scenario by number."""
        preset = get_preset_by_number(preset_number)

        self._state.dataset_kind = preset.dataset_kind
        self._state.round_count = preset.round_count
        self._state.selected_stage = preset.selected_stage
        self._state.min_samples_leaf = preset.min_samples_leaf
        self._state.noise_std = preset.noise_std
        self._state.seed = preset.seed
        self._state.confidence_view_enabled = preset.confidence_view_enabled
        self._state.preset_number = preset_number
        self._state.preset_name = preset.name

        self._rebuild_demo()

    def _apply_next_preset(self) -> None:
        """Cycle to the next preset scenario."""
        next_preset_number = get_next_preset_number(self._state.preset_number)
        self._apply_preset_by_number(next_preset_number)

    def _toggle_dataset(self) -> None:
        """Switch between dataset variants."""
        if self._state.dataset_kind == DATASET_KIND_XOR:
            self._state.dataset_kind = DATASET_KIND_AXIS_ALIGNED
        else:
            self._state.dataset_kind = DATASET_KIND_XOR

        self._mark_custom()
        self._rebuild_demo()

    def _increase_selected_stage(self) -> None:
        """Increase selected boosting stage."""
        self._state.selected_stage = min(
            self._state.round_count,
            self._state.selected_stage + 1,
        )

    def _decrease_selected_stage(self) -> None:
        """Decrease selected boosting stage."""
        self._state.selected_stage = max(
            MIN_ROUND_COUNT,
            self._state.selected_stage - 1,
        )

    def _increase_round_count(self) -> None:
        """Increase total number of boosting rounds."""
        self._state.round_count = min(MAX_ROUND_COUNT, self._state.round_count + 1)
        self._state.selected_stage = self._state.round_count
        self._mark_custom()
        self._rebuild_demo()

    def _decrease_round_count(self) -> None:
        """Decrease total number of boosting rounds."""
        self._state.round_count = max(MIN_ROUND_COUNT, self._state.round_count - 1)
        self._clamp_selected_stage()
        self._mark_custom()
        self._rebuild_demo()

    def _increase_min_samples_leaf(self) -> None:
        """Increase weak learner minimum leaf size."""
        self._state.min_samples_leaf = min(
            MAX_MIN_SAMPLES_LEAF,
            self._state.min_samples_leaf + 1,
        )
        self._mark_custom()
        self._rebuild_demo()

    def _decrease_min_samples_leaf(self) -> None:
        """Decrease weak learner minimum leaf size."""
        self._state.min_samples_leaf = max(
            MIN_MIN_SAMPLES_LEAF,
            self._state.min_samples_leaf - 1,
        )
        self._mark_custom()
        self._rebuild_demo()

    def _increase_noise(self) -> None:
        """Increase dataset noise."""
        self._state.noise_std = min(
            MAX_NOISE_STD,
            round(self._state.noise_std + NOISE_STEP, 2),
        )
        self._mark_custom()
        self._rebuild_demo()

    def _decrease_noise(self) -> None:
        """Decrease dataset noise."""
        self._state.noise_std = max(
            MIN_NOISE_STD,
            round(self._state.noise_std - NOISE_STEP, 2),
        )
        self._mark_custom()
        self._rebuild_demo()

    def _next_seed(self) -> None:
        """Generate next deterministic dataset seed."""
        self._state.seed += 1
        self._mark_custom()
        self._rebuild_demo()

    def _toggle_confidence_view(self) -> None:
        """Toggle confidence-based region coloring."""
        self._state.confidence_view_enabled = not self._state.confidence_view_enabled

    def _reset_defaults(self) -> None:
        """Reset demo controls to default preset."""
        self._apply_preset_by_number(DEFAULT_PRESET_NUMBER)

    def _clamp_selected_stage(self) -> None:
        """Keep selected stage in the valid range."""
        self._state.selected_stage = max(
            MIN_ROUND_COUNT,
            min(self._state.selected_stage, self._state.round_count),
        )

    def _mark_custom(self) -> None:
        """Mark current settings as custom after manual parameter edits."""
        self._state.preset_number = 0
        self._state.preset_name = CUSTOM_PRESET_NAME


def main() -> None:
    """Run the Boosting Mistake Lab Pygame app."""
    app = BoostingPygameApp()
    app.run()
