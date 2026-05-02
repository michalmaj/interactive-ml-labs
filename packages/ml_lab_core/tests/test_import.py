"""Smoke tests for the ml_lab_core package."""


def test_import_ml_lab_core() -> None:
    """The package should be importable."""
    import ml_lab_core

    assert ml_lab_core is not None