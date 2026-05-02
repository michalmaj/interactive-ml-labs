"""Smoke tests for the ml_lab_core package."""
import ml_lab_core


def test_import_ml_lab_core() -> None:
    """The package should be importable."""

    assert ml_lab_core is not None