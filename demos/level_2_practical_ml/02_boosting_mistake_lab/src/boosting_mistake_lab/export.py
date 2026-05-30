"""Decision boundary export utilities for the Boosting Mistake Lab demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from boosting_mistake_lab.boosted_prediction import predict_boosted_ensemble
from boosting_mistake_lab.dataset import WeightedTrainTestDataset
from boosting_mistake_lab.trainer import BoostingTrainerResult

type FloatArray = NDArray[np.float64]

EXPORT_SCHEMA_VERSION: Final[int] = 1
DEFAULT_GRID_RESOLUTION: Final[int] = 40
MIN_GRID_RESOLUTION: Final[int] = 4
MIN_SELECTED_STAGE: Final[int] = 1
WORLD_MARGIN_RATIO: Final[float] = 0.18
MIN_WORLD_SPAN: Final[float] = 1.0
FEATURE_X_INDEX: Final[int] = 0
FEATURE_Y_INDEX: Final[int] = 1


@dataclass(frozen=True, slots=True)
class DecisionBoundaryExportConfig:
    """Configuration for decision boundary export.

    Attributes:
        selected_stage: Boosting stage exported to the decision boundary grid.
        grid_resolution: Number of grid samples per axis.
        include_samples: Whether train/test samples should be included.
    """

    selected_stage: int
    grid_resolution: int = DEFAULT_GRID_RESOLUTION
    include_samples: bool = True


@dataclass(frozen=True, slots=True)
class DecisionBoundaryExportResult:
    """Result of writing a decision boundary export file.

    Attributes:
        output_path: Path to the written JSON file.
        payload: JSON-compatible export payload.
    """

    output_path: Path
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class ExportBounds:
    """World bounds used for decision boundary export."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


def build_decision_boundary_export(
    *,
    dataset: WeightedTrainTestDataset,
    trainer_result: BoostingTrainerResult,
    config: DecisionBoundaryExportConfig,
) -> dict[str, object]:
    """Build JSON-compatible decision boundary export payload.

    Args:
        dataset: Weighted train/test dataset.
        trainer_result: Fitted boosting trainer result.
        config: Export configuration.

    Returns:
        JSON-compatible dictionary.

    Raises:
        ValueError: If export configuration is invalid.
    """
    _validate_config(config=config, round_count=trainer_result.round_count)

    train_features = np.asarray(dataset.train.snapshot.features, dtype=float)
    train_targets = np.asarray(dataset.train.snapshot.targets, dtype=int)
    test_features = np.asarray(dataset.test.snapshot.features, dtype=float)
    test_targets = np.asarray(dataset.test.snapshot.targets, dtype=int)

    bounds = _compute_export_bounds(train_features=train_features, test_features=test_features)

    payload: dict[str, object] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "dataset": _dataset_payload(dataset),
        "selected_stage": config.selected_stage,
        "grid_resolution": config.grid_resolution,
        "bounds": _bounds_payload(bounds),
        "trainer_metrics": _json_ready(trainer_result.snapshot.metrics),
        "staged_history": _staged_history_payload(trainer_result),
        "rounds": _rounds_payload(trainer_result, selected_stage=config.selected_stage),
        "decision_boundary": _decision_boundary_payload(
            trainer_result=trainer_result,
            config=config,
            bounds=bounds,
        ),
    }

    if config.include_samples:
        payload["samples"] = _samples_payload(
            train_features=train_features,
            train_targets=train_targets,
            train_weights=np.asarray(dataset.train.sample_weights, dtype=float),
            final_train_weights=np.asarray(trainer_result.final_train_weights, dtype=float),
            test_features=test_features,
            test_targets=test_targets,
            test_weights=np.asarray(dataset.test.sample_weights, dtype=float),
        )

    return payload


def write_decision_boundary_export(
    *,
    dataset: WeightedTrainTestDataset,
    trainer_result: BoostingTrainerResult,
    config: DecisionBoundaryExportConfig,
    output_path: str | Path,
) -> DecisionBoundaryExportResult:
    """Write decision boundary export payload to a JSON file.

    Args:
        dataset: Weighted train/test dataset.
        trainer_result: Fitted boosting trainer result.
        config: Export configuration.
        output_path: Destination JSON path.

    Returns:
        DecisionBoundaryExportResult with path and payload.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_decision_boundary_export(
        dataset=dataset,
        trainer_result=trainer_result,
        config=config,
    )
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return DecisionBoundaryExportResult(output_path=path, payload=payload)


def _dataset_payload(dataset: WeightedTrainTestDataset) -> dict[str, object]:
    """Build dataset export payload."""
    return {
        "metadata": _json_ready(dataset.metadata),
        "train_sample_count": int(np.asarray(dataset.train.snapshot.features).shape[0]),
        "test_sample_count": int(np.asarray(dataset.test.snapshot.features).shape[0]),
        "feature_names": tuple(dataset.train.snapshot.feature_names),
        "target_names": tuple(dataset.train.snapshot.target_names),
    }


def _staged_history_payload(trainer_result: BoostingTrainerResult) -> dict[str, object]:
    """Build staged history export payload."""
    history = trainer_result.staged_history

    return {
        "round_indices": history.round_indices.tolist(),
        "boosted_train_accuracies": history.boosted_train_accuracies.tolist(),
        "boosted_test_accuracies": history.boosted_test_accuracies.tolist(),
        "boosted_generalization_gaps": history.boosted_generalization_gaps.tolist(),
        "mean_train_confidences": history.mean_train_confidences.tolist(),
        "mean_test_confidences": history.mean_test_confidences.tolist(),
        "learner_weights": history.learner_weights.tolist(),
        "weighted_train_errors": history.weighted_train_errors.tolist(),
    }


def _rounds_payload(
    trainer_result: BoostingTrainerResult,
    *,
    selected_stage: int,
) -> list[dict[str, object]]:
    """Build weak learner round payloads up to selected stage."""
    payloads: list[dict[str, object]] = []

    for round_result in trainer_result.round_results[:selected_stage]:
        split = round_result.weak_learner.split

        payloads.append(
            {
                "round_index": round_result.round_index,
                "learner_weight": round_result.learner_weight,
                "weighted_train_error": round_result.weighted_train_error,
                "train_accuracy": float(
                    round_result.round_snapshot.metrics["train_accuracy"],
                ),
                "test_accuracy": float(
                    round_result.round_snapshot.metrics["test_accuracy"],
                ),
                "split": {
                    "feature_index": split.feature_index,
                    "threshold": split.threshold,
                    "left_prediction": split.left_prediction,
                    "right_prediction": split.right_prediction,
                    "training_error": split.training_error,
                    "weighted_training_error": split.weighted_training_error,
                },
            },
        )

    return payloads


def _decision_boundary_payload(
    *,
    trainer_result: BoostingTrainerResult,
    config: DecisionBoundaryExportConfig,
    bounds: ExportBounds,
) -> dict[str, object]:
    """Build decision boundary grid payload."""
    x_values = np.linspace(bounds.x_min, bounds.x_max, config.grid_resolution)
    y_values = np.linspace(bounds.y_min, bounds.y_max, config.grid_resolution)
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    prediction = _predict_selected_stage(
        trainer_result=trainer_result,
        selected_stage=config.selected_stage,
        features=grid_points,
    )
    shape = (config.grid_resolution, config.grid_resolution)

    return {
        "x_values": x_values.tolist(),
        "y_values": y_values.tolist(),
        "predictions": prediction.predictions.reshape(shape).tolist(),
        "confidence": prediction.confidence.reshape(shape).tolist(),
        "raw_scores": prediction.raw_scores.reshape(shape).tolist(),
        "normalized_margins": prediction.normalized_margins.reshape(shape).tolist(),
    }


def _predict_selected_stage(
    *,
    trainer_result: BoostingTrainerResult,
    selected_stage: int,
    features: FloatArray,
):
    """Predict with boosted ensemble up to selected stage."""
    selected_rounds = trainer_result.round_results[:selected_stage]

    return predict_boosted_ensemble(
        weak_learners=[round_result.weak_learner for round_result in selected_rounds],
        learner_weights=trainer_result.learner_weights[:selected_stage],
        features=features,
    )


def _samples_payload(
    *,
    train_features: FloatArray,
    train_targets: NDArray[np.int_],
    train_weights: FloatArray,
    final_train_weights: FloatArray,
    test_features: FloatArray,
    test_targets: NDArray[np.int_],
    test_weights: FloatArray,
) -> dict[str, object]:
    """Build sample payload."""
    return {
        "train": {
            "features": train_features.tolist(),
            "targets": train_targets.tolist(),
            "initial_weights": train_weights.tolist(),
            "final_weights": final_train_weights.tolist(),
        },
        "test": {
            "features": test_features.tolist(),
            "targets": test_targets.tolist(),
            "weights": test_weights.tolist(),
        },
    }


def _compute_export_bounds(
    *,
    train_features: FloatArray,
    test_features: FloatArray,
) -> ExportBounds:
    """Compute decision boundary export bounds."""
    features = np.vstack([train_features, test_features])

    x_min = float(np.min(features[:, FEATURE_X_INDEX]))
    x_max = float(np.max(features[:, FEATURE_X_INDEX]))
    y_min = float(np.min(features[:, FEATURE_Y_INDEX]))
    y_max = float(np.max(features[:, FEATURE_Y_INDEX]))

    x_span = max(x_max - x_min, MIN_WORLD_SPAN)
    y_span = max(y_max - y_min, MIN_WORLD_SPAN)

    return ExportBounds(
        x_min=x_min - x_span * WORLD_MARGIN_RATIO,
        x_max=x_max + x_span * WORLD_MARGIN_RATIO,
        y_min=y_min - y_span * WORLD_MARGIN_RATIO,
        y_max=y_max + y_span * WORLD_MARGIN_RATIO,
    )


def _bounds_payload(bounds: ExportBounds) -> dict[str, float]:
    """Build bounds payload."""
    return {
        "x_min": bounds.x_min,
        "x_max": bounds.x_max,
        "y_min": bounds.y_min,
        "y_max": bounds.y_max,
    }


def _json_ready(value: object) -> object:
    """Convert common NumPy/Python values into JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}

    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]

    return value


def _validate_config(
    *,
    config: DecisionBoundaryExportConfig,
    round_count: int,
) -> None:
    """Validate export configuration."""
    if config.selected_stage < MIN_SELECTED_STAGE:
        msg = "selected_stage must be greater than or equal to 1."
        raise ValueError(msg)

    if config.selected_stage > round_count:
        msg = (
            "selected_stage cannot be greater than completed round count. "
            f"Got {config.selected_stage} and {round_count}."
        )
        raise ValueError(msg)

    if config.grid_resolution < MIN_GRID_RESOLUTION:
        msg = f"grid_resolution must be greater than or equal to {MIN_GRID_RESOLUTION}."
        raise ValueError(msg)
