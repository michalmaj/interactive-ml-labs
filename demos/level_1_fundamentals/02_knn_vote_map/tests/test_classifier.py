"""Tests for the k-nearest neighbors classifier."""

import numpy as np
import pytest
from knn_vote_map import (
    KNearestNeighborsClassifier,
    KNearestNeighborsConfig,
    make_synthetic_classification_dataset,
)
from ml_lab_core import AlgorithmSnapshot, DatasetSnapshot

K_ONE: int = 1
K_THREE: int = 3
K_TOO_LARGE: int = 10


def _simple_dataset() -> DatasetSnapshot:
    """Create a small deterministic classification dataset."""
    return DatasetSnapshot(
        features=np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [5.0, 5.0],
                [5.0, 6.0],
                [6.0, 5.0],
            ],
            dtype=float,
        ),
        targets=np.array([0, 0, 0, 1, 1, 1], dtype=int),
        feature_names=("x1", "x2"),
        target_names=("class_0", "class_1"),
    )


def test_knn_classifier_can_fit_dataset() -> None:
    """Classifier should store training data after fitting."""
    classifier = KNearestNeighborsClassifier()
    dataset = _simple_dataset()

    classifier.fit(dataset)

    assert classifier.is_fitted is True


def test_knn_classifier_predicts_nearest_class_for_k_one() -> None:
    """For k=1, the prediction should use the nearest single neighbor."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_ONE))
    classifier.fit(_simple_dataset())

    result = classifier.predict_one([0.2, 0.1])

    assert result.predicted_label == 0
    assert len(result.neighbors) == K_ONE
    assert result.neighbors[0].label == 0


def test_knn_classifier_predicts_by_majority_vote() -> None:
    """For k>1, the prediction should use majority voting."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_THREE))
    classifier.fit(_simple_dataset())

    result = classifier.predict_one([5.2, 5.1])

    assert result.predicted_label == 1
    assert len(result.neighbors) == K_THREE
    assert result.vote_counts == {1: 3}


def test_knn_classifier_resolves_ties_by_smallest_label() -> None:
    """Tie voting should be deterministic."""
    dataset = DatasetSnapshot(
        features=np.array(
            [
                [-1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=float,
        ),
        targets=np.array([0, 1], dtype=int),
    )
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=2))
    classifier.fit(dataset)

    result = classifier.predict_one([0.0, 0.0])

    assert result.vote_counts == {0: 1, 1: 1}
    assert result.predicted_label == 0


def test_knn_classifier_predicts_many_points() -> None:
    """Classifier should predict labels for multiple query points."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_THREE))
    classifier.fit(_simple_dataset())

    predictions = classifier.predict(
        np.array(
            [
                [0.1, 0.1],
                [5.1, 5.1],
            ],
        ),
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1]))


def test_knn_classifier_predict_accepts_single_point() -> None:
    """Predict should also accept a single query point."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_THREE))
    classifier.fit(_simple_dataset())

    predictions = classifier.predict([0.1, 0.1])

    np.testing.assert_array_equal(predictions, np.array([0]))


def test_knn_classifier_snapshot_contains_prediction_details() -> None:
    """Snapshot should expose latest prediction details for renderers."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_THREE))
    classifier.fit(_simple_dataset())

    classifier.predict_one([0.1, 0.1])
    snapshot = classifier.snapshot()

    assert isinstance(snapshot, AlgorithmSnapshot)
    assert snapshot.status == "ready"
    assert snapshot.metrics["k"] == K_THREE
    assert snapshot.metrics["predicted_label"] == 0
    assert snapshot.visual_state["query_point"] is not None
    assert snapshot.visual_state["neighbors"] is not None
    assert snapshot.visual_state["vote_counts"] == {0: 3}


def test_knn_classifier_rejects_prediction_before_fit() -> None:
    """Prediction before fit should fail clearly."""
    classifier = KNearestNeighborsClassifier()

    with pytest.raises(RuntimeError, match="fitted"):
        classifier.predict_one([0.0, 0.0])


def test_knn_classifier_rejects_invalid_k() -> None:
    """k must be positive."""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        KNearestNeighborsClassifier(KNearestNeighborsConfig(k=0))


def test_knn_classifier_rejects_k_larger_than_dataset() -> None:
    """k cannot be greater than the number of training samples."""
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_TOO_LARGE))

    with pytest.raises(ValueError, match="k cannot be greater"):
        classifier.fit(_simple_dataset())


@pytest.mark.parametrize(
    "dataset, expected_message",
    [
        (
            DatasetSnapshot(features=np.array([1.0, 2.0, 3.0]), targets=np.array([0, 1, 0])),
            "two-dimensional array",
        ),
        (
            DatasetSnapshot(features=np.empty((0, 2)), targets=np.array([])),
            "cannot be empty",
        ),
        (
            DatasetSnapshot(features=np.empty((3, 0)), targets=np.array([0, 1, 0])),
            "at least one feature",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0]]), targets=None),
            "targets are required",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0]]), targets=np.array([[0]])),
            "one-dimensional array",
        ),
        (
            DatasetSnapshot(features=np.array([[1.0, 2.0], [3.0, 4.0]]), targets=np.array([0])),
            "same number of samples",
        ),
    ],
)
def test_knn_classifier_rejects_invalid_dataset(
    dataset: DatasetSnapshot,
    expected_message: str,
) -> None:
    """Invalid datasets should fail clearly."""
    classifier = KNearestNeighborsClassifier()

    with pytest.raises(ValueError, match=expected_message):
        classifier.fit(dataset)


@pytest.mark.parametrize(
    "query_point, expected_message",
    [
        (
            np.array([[0.0, 0.0]]),
            "one-dimensional",
        ),
        (
            np.array([]),
            "cannot be empty",
        ),
        (
            np.array([0.0, 0.0, 0.0]),
            "same number of features",
        ),
    ],
)
def test_knn_classifier_rejects_invalid_query_point(
    query_point: np.ndarray,
    expected_message: str,
) -> None:
    """Invalid query points should fail clearly."""
    classifier = KNearestNeighborsClassifier()
    classifier.fit(_simple_dataset())

    with pytest.raises(ValueError, match=expected_message):
        classifier.predict_one(query_point)


def test_knn_classifier_works_with_synthetic_dataset() -> None:
    """Classifier should work with the generated synthetic dataset."""
    dataset = make_synthetic_classification_dataset()
    classifier = KNearestNeighborsClassifier(KNearestNeighborsConfig(k=K_THREE))

    classifier.fit(dataset)
    result = classifier.predict_one([0.0, 0.0])

    assert result.predicted_label in {0, 1}
    assert len(result.neighbors) == K_THREE
