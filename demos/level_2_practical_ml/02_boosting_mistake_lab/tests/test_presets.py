"""Tests for Boosting Mistake Lab preset scenarios."""

import pytest
from boosting_mistake_lab import (
    DATASET_KIND_AXIS_ALIGNED,
    DATASET_KIND_XOR,
    DEFAULT_PRESET_NUMBER,
    BoostingPreset,
    available_presets,
    get_next_preset_number,
    get_preset_by_number,
    preset_count,
)


def test_available_presets_returns_presets() -> None:
    """Available presets should return non-empty preset collection."""
    presets = available_presets()

    assert presets
    assert all(isinstance(preset, BoostingPreset) for preset in presets)


def test_preset_count_matches_available_presets() -> None:
    """Preset count should match available preset collection length."""
    assert preset_count() == len(available_presets())


def test_default_preset_exists() -> None:
    """Default preset number should point to a valid preset."""
    preset = get_preset_by_number(DEFAULT_PRESET_NUMBER)

    assert isinstance(preset, BoostingPreset)


def test_presets_have_valid_dataset_kinds() -> None:
    """Preset dataset kinds should be supported."""
    for preset in available_presets():
        assert preset.dataset_kind in {DATASET_KIND_AXIS_ALIGNED, DATASET_KIND_XOR}


def test_presets_have_valid_round_settings() -> None:
    """Preset round and selected-stage settings should be valid."""
    for preset in available_presets():
        assert preset.round_count >= 1
        assert preset.selected_stage >= 1
        assert preset.selected_stage <= preset.round_count


def test_presets_have_valid_model_settings() -> None:
    """Preset model settings should be valid."""
    for preset in available_presets():
        assert preset.min_samples_leaf >= 1
        assert preset.noise_std >= 0.0


@pytest.mark.parametrize("preset_number", [1, 2, 3, 4])
def test_get_preset_by_number_returns_expected_preset(preset_number: int) -> None:
    """Preset lookup should use one-based preset numbers."""
    preset = get_preset_by_number(preset_number)

    assert preset == available_presets()[preset_number - 1]


@pytest.mark.parametrize("preset_number", [0, -1, 999])
def test_get_preset_by_number_rejects_invalid_number(preset_number: int) -> None:
    """Invalid preset numbers should fail clearly."""
    with pytest.raises(ValueError, match="preset_number"):
        get_preset_by_number(preset_number)


def test_get_next_preset_number_wraps_around() -> None:
    """Next preset number should wrap from last preset to first."""
    assert get_next_preset_number(preset_count()) == 1


def test_get_next_preset_number_handles_custom_state() -> None:
    """Custom state should continue with first preset."""
    assert get_next_preset_number(0) == 1
