"""Tests for metrics history utilities."""

import pytest
from ml_lab_core import MetricsHistory


def test_metrics_history_is_empty_by_default() -> None:
    """MetricsHistory should not contain values after initialization."""
    history = MetricsHistory()

    assert history.is_empty()
    assert len(history) == 0
    assert history.names() == ()


def test_metrics_history_records_metric_values() -> None:
    """MetricsHistory should store numeric values for a metric."""
    history = MetricsHistory()

    history.add("loss", 1.0)
    history.add("loss", 0.5)

    assert history.latest("loss") == 0.5
    assert history.series("loss") == (1.0, 0.5)
    assert len(history) == 2


def test_metrics_history_tracks_multiple_metrics() -> None:
    """MetricsHistory should store independent series for multiple metrics."""
    history = MetricsHistory()

    history.add("loss", 1.0)
    history.add("accuracy", 0.75)
    history.add("accuracy", 0.8)

    assert history.names() == ("loss", "accuracy")
    assert history.latest("loss") == 1.0
    assert history.latest("accuracy") == 0.8
    assert len(history) == 3


def test_metrics_history_returns_empty_series_for_unknown_metric() -> None:
    """Unknown metric series should be represented by an empty tuple."""
    history = MetricsHistory()

    assert history.series("unknown") == ()


def test_metrics_history_raises_for_missing_latest_metric() -> None:
    """Requesting the latest value for an unknown metric should fail clearly."""
    history = MetricsHistory()

    with pytest.raises(KeyError, match="accuracy"):
        history.latest("accuracy")


def test_metrics_history_rejects_empty_metric_name() -> None:
    """Metric names should not be empty."""
    history = MetricsHistory()

    with pytest.raises(ValueError, match="Metric name cannot be empty"):
        history.add("", 1.0)


def test_metrics_history_returns_copy_as_dict() -> None:
    """The dictionary representation should not expose internal lists."""
    history = MetricsHistory()
    history.add("loss", 1.0)
    history.add("loss", 0.5)

    values = history.as_dict()

    assert values == {"loss": (1.0, 0.5)}


def test_metrics_history_can_be_reset() -> None:
    """Reset should remove all stored metric values."""
    history = MetricsHistory()
    history.add("loss", 1.0)

    history.reset()

    assert history.is_empty()
    assert len(history) == 0
    assert history.series("loss") == ()
