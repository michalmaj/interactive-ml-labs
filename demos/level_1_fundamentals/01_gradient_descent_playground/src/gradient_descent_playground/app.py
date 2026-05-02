"""Command-line entry point for the Gradient Descent Playground demo."""

from __future__ import annotations

from gradient_descent_playground.algorithm import (
    GradientDescentConfig,
    StepwiseLinearRegression,
)
from gradient_descent_playground.dataset import make_synthetic_regression_dataset


def main() -> None:
    """Run a minimal command-line version of the demo.

    The interactive Pygame implementation will be added in later pull requests.
    This CLI version already demonstrates real stepwise learning.
    """
    dataset = make_synthetic_regression_dataset()
    model = StepwiseLinearRegression(
        GradientDescentConfig(
            learning_rate=0.03,
            max_steps=10,
        ),
    )

    model.reset(dataset)

    print("Gradient Descent Playground")
    print("Step, Loss, Weight, Bias")

    snapshot = model.snapshot()
    print(
        f"{snapshot.iteration}, "
        f"{snapshot.metrics['loss']:.6f}, "
        f"{snapshot.metrics['weight']:.6f}, "
        f"{snapshot.metrics['bias']:.6f}",
    )

    while not snapshot.done:
        snapshot = model.step()
        print(
            f"{snapshot.iteration}, "
            f"{snapshot.metrics['loss']:.6f}, "
            f"{snapshot.metrics['weight']:.6f}, "
            f"{snapshot.metrics['bias']:.6f}",
        )
