"""Utilities for storing metric values over time."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricsHistory:
    """Store numeric metric values collected over multiple algorithm steps.

    The class is intentionally small and independent from plotting libraries.
    Renderers and demos can use it to visualize loss curves, accuracy changes,
    rewards, inertia, or any other numeric metric.

    Examples:
        >>> history = MetricsHistory()
        >>> history.add("loss", 1.5)
        >>> history.add("loss", 0.8)
        >>> history.latest("loss")
        0.8
        >>> history.series("loss")
        (1.5, 0.8)
    """

    _values: dict[str, list[float]] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        """Record one value for a metric.

        Args:
            name: Metric name, for example ``\"loss\"`` or ``\"accuracy\"``.
            value: Numeric metric value.

        Raises:
            ValueError: If the metric name is empty.
        """
        if not name:
            msg = "Metric name cannot be empty."
            raise ValueError(msg)

        self._values.setdefault(name, []).append(float(value))

    def latest(self, name: str) -> float:
        """Return the latest value recorded for a metric.

        Args:
            name: Metric name.

        Raises:
            KeyError: If the metric does not exist or has no values.
        """
        values = self._values.get(name)

        if not values:
            msg = f"No values recorded for metric: {name}"
            raise KeyError(msg)

        return values[-1]

    def series(self, name: str) -> tuple[float, ...]:
        """Return all values recorded for a metric.

        Unknown metrics return an empty tuple. This is convenient for renderers,
        where missing metrics usually mean that nothing should be drawn yet.
        """
        return tuple(self._values.get(name, ()))

    def names(self) -> tuple[str, ...]:
        """Return names of all recorded metrics."""
        return tuple(self._values)

    def as_dict(self) -> dict[str, tuple[float, ...]]:
        """Return a copy of the full metric history."""
        return {name: tuple(values) for name, values in self._values.items()}

    def reset(self) -> None:
        """Remove all recorded metric values."""
        self._values.clear()

    def is_empty(self) -> bool:
        """Return whether no metric values have been recorded."""
        return not self._values

    def __len__(self) -> int:
        """Return the total number of recorded metric values."""
        return sum(len(values) for values in self._values.values())
