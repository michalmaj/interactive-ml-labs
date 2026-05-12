"""Command-line entry point for the Boosting Mistake Lab demo."""

from __future__ import annotations

from ml_lab_core import MetricsHistory


def main() -> None:
    """Run a minimal placeholder version of the demo.

    The real boosting implementation will be added in later pull requests.
    This entry point verifies that the package is connected to the workspace
    and can use shared utilities from `ml_lab_core`.
    """
    history = MetricsHistory()
    history.add("placeholder_round_count", 5.0)
    history.add("placeholder_weak_learner_error", 0.35)
    history.add("placeholder_sample_weight_sum", 1.0)

    print("Boosting Mistake Lab")
    print("Package skeleton is configured correctly.")
    print(f"Placeholder boosting rounds: {history.latest('placeholder_round_count'):.0f}")
    print(
        f"Placeholder weak learner error: {history.latest('placeholder_weak_learner_error'):.2f}",
    )
    print(
        f"Placeholder sample weight sum: {history.latest('placeholder_sample_weight_sum'):.2f}",
    )
