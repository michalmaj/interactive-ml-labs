"""Command-line entry point for the Random Forest Bagging Lab demo."""

from __future__ import annotations

from ml_lab_core import MetricsHistory


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real random forest implementation will be added in later pull requests.
    This entry point verifies that the package is correctly connected to the
    workspace and can use shared utilities from `ml_lab_core`.
    """
    history = MetricsHistory()
    history.add("placeholder_tree_count", 10.0)
    history.add("placeholder_bootstrap_ratio", 0.8)
    history.add("placeholder_vote_confidence", 0.75)

    print("Random Forest Bagging Lab")
    print("Package skeleton is configured correctly.")
    print(f"Placeholder tree count: {history.latest('placeholder_tree_count'):.0f}")
    print(f"Placeholder bootstrap ratio: {history.latest('placeholder_bootstrap_ratio'):.2f}")
    print(f"Placeholder vote confidence: {history.latest('placeholder_vote_confidence'):.2f}")