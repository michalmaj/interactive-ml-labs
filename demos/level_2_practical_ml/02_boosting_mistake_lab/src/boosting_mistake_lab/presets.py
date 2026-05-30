"""Preset scenarios for the Boosting Mistake Lab demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from boosting_mistake_lab.dataset import DATASET_KIND_AXIS_ALIGNED, DATASET_KIND_XOR

CUSTOM_PRESET_NAME: Final[str] = "Custom"
DEFAULT_PRESET_NUMBER: Final[int] = 2

MIN_PRESET_NUMBER: Final[int] = 1
MIN_ROUND_COUNT: Final[int] = 1
MIN_SELECTED_STAGE: Final[int] = 1
MIN_MIN_SAMPLES_LEAF: Final[int] = 1
MIN_NOISE_STD: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class BoostingPreset:
    """Ready-to-use Pygame scenario.

    Attributes:
        name: Human-readable preset name.
        description: Short teaching-oriented description.
        dataset_kind: Dataset variant.
        round_count: Total number of boosting rounds.
        selected_stage: Initially selected boosting stage.
        min_samples_leaf: Weak learner minimum leaf size.
        noise_std: Dataset noise level.
        seed: Dataset random seed.
        confidence_view_enabled: Whether confidence view starts enabled.
    """

    name: str
    description: str
    dataset_kind: str
    round_count: int
    selected_stage: int
    min_samples_leaf: int
    noise_std: float
    seed: int
    confidence_view_enabled: bool = True


PRESETS: Final[tuple[BoostingPreset, ...]] = (
    BoostingPreset(
        name="Easy axis-aligned",
        description="Simple baseline where a single stump already performs well.",
        dataset_kind=DATASET_KIND_AXIS_ALIGNED,
        round_count=4,
        selected_stage=4,
        min_samples_leaf=1,
        noise_std=0.25,
        seed=42,
        confidence_view_enabled=True,
    ),
    BoostingPreset(
        name="Noisy XOR",
        description="Nonlinear dataset where boosted stumps are more interesting.",
        dataset_kind=DATASET_KIND_XOR,
        round_count=8,
        selected_stage=8,
        min_samples_leaf=1,
        noise_std=0.65,
        seed=42,
        confidence_view_enabled=True,
    ),
    BoostingPreset(
        name="Overfitting watch",
        description="More rounds and noise make staged accuracy worth inspecting.",
        dataset_kind=DATASET_KIND_XOR,
        round_count=12,
        selected_stage=12,
        min_samples_leaf=1,
        noise_std=0.90,
        seed=18,
        confidence_view_enabled=True,
    ),
    BoostingPreset(
        name="Low-round challenge",
        description="Small round budget for discussing accuracy-complexity trade-offs.",
        dataset_kind=DATASET_KIND_XOR,
        round_count=4,
        selected_stage=4,
        min_samples_leaf=2,
        noise_std=0.55,
        seed=7,
        confidence_view_enabled=True,
    ),
)


def available_presets() -> tuple[BoostingPreset, ...]:
    """Return all available presets."""
    return PRESETS


def preset_count() -> int:
    """Return number of available presets."""
    return len(PRESETS)


def get_preset_by_number(preset_number: int) -> BoostingPreset:
    """Return preset by one-based number.

    Args:
        preset_number: One-based preset number.

    Returns:
        BoostingPreset.

    Raises:
        ValueError: If preset number is outside the available range.
    """
    if preset_number < MIN_PRESET_NUMBER or preset_number > preset_count():
        msg = (
            "preset_number must be in the available preset range. "
            f"Got {preset_number}; expected 1..{preset_count()}."
        )
        raise ValueError(msg)

    preset = PRESETS[preset_number - 1]
    _validate_preset(preset)

    return preset


def get_next_preset_number(current_preset_number: int) -> int:
    """Return next preset number, wrapping around.

    Args:
        current_preset_number: Current one-based preset number.

    Returns:
        Next one-based preset number.
    """
    if current_preset_number < MIN_PRESET_NUMBER or current_preset_number > preset_count():
        return MIN_PRESET_NUMBER

    return current_preset_number % preset_count() + 1


def _validate_preset(preset: BoostingPreset) -> None:
    """Validate preset values."""
    if preset.dataset_kind not in {DATASET_KIND_AXIS_ALIGNED, DATASET_KIND_XOR}:
        msg = f"Unsupported dataset kind: {preset.dataset_kind}."
        raise ValueError(msg)

    if preset.round_count < MIN_ROUND_COUNT:
        msg = "round_count must be greater than or equal to 1."
        raise ValueError(msg)

    if preset.selected_stage < MIN_SELECTED_STAGE:
        msg = "selected_stage must be greater than or equal to 1."
        raise ValueError(msg)

    if preset.selected_stage > preset.round_count:
        msg = "selected_stage cannot be greater than round_count."
        raise ValueError(msg)

    if preset.min_samples_leaf < MIN_MIN_SAMPLES_LEAF:
        msg = "min_samples_leaf must be greater than or equal to 1."
        raise ValueError(msg)

    if preset.noise_std < MIN_NOISE_STD:
        msg = "noise_std cannot be negative."
        raise ValueError(msg)
