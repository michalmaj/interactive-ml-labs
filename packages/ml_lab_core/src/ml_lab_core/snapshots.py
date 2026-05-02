"""Shared snapshot objects used by interactive machine learning demos."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MetricValue: type = int | float | str | bool
Metrics: type = Mapping[str, MetricValue]
VisualState: type = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """Container describing the current dataset used by a demo.

    The core package intentionally does not require NumPy, pandas, or scikit-learn.
    Demo packages may store arrays, tensors, data frames, or custom objects in
    `features` and `targets`.

    Attributes:
        features: Input data used by the algorithm.
        targets: Optional target values or labels.
        feature_names: Optional names of input features.
        target_names: Optional names of target classes or outputs.
        metadata: Additional dataset information useful for demos.
    """

    features: Any
    targets: Any | None = None
    feature_names: tuple[str, ...] = ()
    target_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlgorithmSnapshot:
    """Container describing the current state of a stepwise algorithm.

    A snapshot is the communication boundary between the algorithm, metrics,
    explanation panels, and renderers.

    The algorithm produces snapshots. The renderer consumes snapshots.
    This keeps machine learning logic independent from the UI layer.

    Attributes:
        iteration: Current algorithm iteration or step number.
        status: Human-readable algorithm status.
        visual_state: Data required by renderers.
        metrics: Live metrics such as loss, accuracy, inertia, or reward.
        annotations: Short textual notes explaining what happened.
        done: Whether the algorithm has finished.
        success: Optional challenge-oriented success flag.
    """

    iteration: int = 0
    status: str = "initialized"
    visual_state: VisualState = field(default_factory=dict)
    metrics: Metrics = field(default_factory=dict)
    annotations: tuple[str, ...] = ()
    done: bool = False
    success: bool | None = None
