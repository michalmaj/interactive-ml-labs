"""Tests for Boosting Mistake Lab decision boundary export."""

import json

import pytest
from boosting_mistake_lab import (
    DATASET_KIND_XOR,
    BoostingTrainer,
    BoostingTrainerConfig,
    DecisionBoundaryExportConfig,
    DecisionBoundaryExportResult,
    SyntheticWeightedDatasetConfig,
    build_decision_boundary_export,
    make_synthetic_weighted_dataset,
    write_decision_boundary_export,
)

SAMPLES_PER_CLASS: int = 12
CLASS_DISTANCE: float = 6.0
NOISE_STD_ZERO: float = 0.0
SEED: int = 123
ROUND_COUNT: int = 4
SELECTED_STAGE: int = 3
GRID_RESOLUTION: int = 12


def _dataset_and_result():
    """Create deterministic dataset and trainer result."""
    dataset = make_synthetic_weighted_dataset(
        SyntheticWeightedDatasetConfig(
            train_samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=SAMPLES_PER_CLASS,
            class_distance=CLASS_DISTANCE,
            noise_std=NOISE_STD_ZERO,
            seed=SEED,
            dataset_kind=DATASET_KIND_XOR,
        ),
    )
    trainer = BoostingTrainer(BoostingTrainerConfig(round_count=ROUND_COUNT))
    result = trainer.reset(dataset)

    return dataset, result


def test_build_decision_boundary_export_returns_payload() -> None:
    """Decision boundary export builder should return a payload."""
    dataset, result = _dataset_and_result()

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
        ),
    )

    assert payload["schema_version"] == 1
    assert payload["selected_stage"] == SELECTED_STAGE
    assert payload["grid_resolution"] == GRID_RESOLUTION


def test_build_decision_boundary_export_contains_core_sections() -> None:
    """Decision boundary export should contain expected sections."""
    dataset, result = _dataset_and_result()

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
        ),
    )

    assert "dataset" in payload
    assert "bounds" in payload
    assert "trainer_metrics" in payload
    assert "staged_history" in payload
    assert "rounds" in payload
    assert "decision_boundary" in payload
    assert "samples" in payload


def test_build_decision_boundary_export_uses_selected_stage_rounds() -> None:
    """Exported round list should include rounds up to selected stage."""
    dataset, result = _dataset_and_result()

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
        ),
    )

    rounds = payload["rounds"]

    assert isinstance(rounds, list)
    assert len(rounds) == SELECTED_STAGE
    assert rounds[-1]["round_index"] == SELECTED_STAGE


def test_build_decision_boundary_export_boundary_grid_has_expected_shape() -> None:
    """Decision boundary grid should have grid_resolution by grid_resolution shape."""
    dataset, result = _dataset_and_result()

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
        ),
    )
    boundary = payload["decision_boundary"]

    assert isinstance(boundary, dict)
    assert len(boundary["x_values"]) == GRID_RESOLUTION
    assert len(boundary["y_values"]) == GRID_RESOLUTION
    assert len(boundary["predictions"]) == GRID_RESOLUTION
    assert len(boundary["predictions"][0]) == GRID_RESOLUTION
    assert len(boundary["confidence"]) == GRID_RESOLUTION
    assert len(boundary["confidence"][0]) == GRID_RESOLUTION


def test_write_decision_boundary_export_writes_json_file(tmp_path) -> None:
    """Decision boundary export should be written as JSON."""
    dataset, result = _dataset_and_result()
    output_path = tmp_path / "boundary.json"

    export_result = write_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
        ),
        output_path=output_path,
    )

    assert isinstance(export_result, DecisionBoundaryExportResult)
    assert export_result.output_path == output_path
    assert output_path.exists()

    loaded_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded_payload["selected_stage"] == SELECTED_STAGE
    assert loaded_payload["grid_resolution"] == GRID_RESOLUTION


def test_build_decision_boundary_export_can_skip_samples() -> None:
    """Sample payload should be optional."""
    dataset, result = _dataset_and_result()

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=result,
        config=DecisionBoundaryExportConfig(
            selected_stage=SELECTED_STAGE,
            grid_resolution=GRID_RESOLUTION,
            include_samples=False,
        ),
    )

    assert "samples" not in payload


@pytest.mark.parametrize(
    "config, expected_message",
    [
        (
            DecisionBoundaryExportConfig(selected_stage=0),
            "selected_stage",
        ),
        (
            DecisionBoundaryExportConfig(selected_stage=ROUND_COUNT + 1),
            "selected_stage",
        ),
        (
            DecisionBoundaryExportConfig(
                selected_stage=SELECTED_STAGE,
                grid_resolution=2,
            ),
            "grid_resolution",
        ),
    ],
)
def test_build_decision_boundary_export_rejects_invalid_config(
    config: DecisionBoundaryExportConfig,
    expected_message: str,
) -> None:
    """Invalid export config should fail clearly."""
    dataset, result = _dataset_and_result()

    with pytest.raises(ValueError, match=expected_message):
        build_decision_boundary_export(
            dataset=dataset,
            trainer_result=result,
            config=config,
        )
