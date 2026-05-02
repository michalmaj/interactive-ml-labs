"""Command-line entry point for the Gradient Descent Playground demo."""

from __future__ import annotations

from ml_lab_core import MetricsHistory


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point exists to verify that the demo package is correctly
    connected to the workspace and can use `ml_lab_core`.
    """
    history = MetricsHistory()
    history.add("loss", 1.0)
    history.add("loss", 0.5)

    print("Gradient Descent Playground")
    print("Package skeleton is configured correctly.")
    print(f"Latest placeholder loss: {history.latest('loss')}")
