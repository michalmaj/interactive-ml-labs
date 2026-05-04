"""Command-line entry point for the k-NN Vote Map demo."""

from __future__ import annotations

from ml_lab_core import MetricsHistory


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point verifies that the package is correctly connected to the
    workspace and can use shared utilities from `ml_lab_core`.
    """
    history = MetricsHistory()
    history.add("placeholder_accuracy", 1.0)

    print("k-NN Vote Map")
    print("Package skeleton is configured correctly.")
    print(f"Placeholder accuracy: {history.latest('placeholder_accuracy'):.2f}")
