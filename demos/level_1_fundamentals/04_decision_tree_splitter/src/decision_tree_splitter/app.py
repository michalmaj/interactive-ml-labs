"""Command-line entry point for the Decision Tree Splitter demo."""

from __future__ import annotations

from ml_lab_core import MetricsHistory


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real interactive implementation will be added in later pull requests.
    This entry point verifies that the package is correctly connected to the
    workspace and can use shared utilities from `ml_lab_core`.
    """
    history = MetricsHistory()
    history.add("placeholder_gini", 0.5)
    history.add("placeholder_depth", 0.0)

    print("Decision Tree Splitter")
    print("Package skeleton is configured correctly.")
    print(f"Placeholder Gini impurity: {history.latest('placeholder_gini'):.2f}")
    print(f"Placeholder tree depth: {history.latest('placeholder_depth'):.0f}")
