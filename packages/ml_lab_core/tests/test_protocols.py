"""Tests for algorithm protocols."""

from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot, StepwiseAlgorithm


class DummyAlgorithm:
    """Minimal algorithm used to verify the StepwiseAlgorithm protocol."""

    name = "dummy"

    def __init__(self) -> None:
        """Initialize the dummy algorithm."""
        self._iteration = 0

    def reset(self, dataset: DatasetSnapshot) -> None:
        """Reset the dummy algorithm."""
        self._iteration = 0
        self._dataset = dataset

    def step(self) -> AlgorithmSnapshot:
        """Advance the dummy algorithm by one step."""
        self._iteration += 1
        return self.snapshot()

    def snapshot(self) -> AlgorithmSnapshot:
        """Return the current dummy state."""
        return AlgorithmSnapshot(
            iteration=self._iteration,
            status="running",
            metrics={"iteration": self._iteration},
        )


def test_dummy_algorithm_matches_stepwise_protocol() -> None:
    """A class with the expected methods should satisfy StepwiseAlgorithm."""
    algorithm = DummyAlgorithm()

    assert isinstance(algorithm, StepwiseAlgorithm)


def test_dummy_algorithm_returns_snapshots() -> None:
    """Stepwise algorithms should return AlgorithmSnapshot instances."""
    algorithm = DummyAlgorithm()
    dataset = DatasetSnapshot(features=[1, 2, 3])

    algorithm.reset(dataset)
    snapshot = algorithm.step()

    assert snapshot.iteration == 1
    assert snapshot.status == "running"
    assert snapshot.metrics["iteration"] == 1
