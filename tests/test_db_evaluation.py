"""Independent squared-distance DB oracles for the shared evaluator."""

import numpy as np
import pytest

from src.cvi import DB


def direct_db(points, labels):
    """Compute DB from the samples, without reading CVI summary statistics."""
    groups = [points[labels == label] for label in dict.fromkeys(labels)]
    centers = [group.mean(axis=0) for group in groups]
    scatter = [np.mean(np.sum((group - center) ** 2, axis=1))
               for group, center in zip(groups, centers)]
    return np.mean([
        max((scatter[i] + scatter[j]) / np.sum((centers[i] - centers[j]) ** 2)
            for j in range(len(groups)) if j != i)
        for i in range(len(groups))
    ])


@pytest.mark.parametrize("n_clusters", [2, 10, 128])
def test_shared_db_evaluator_matches_sample_definition(backend, n_clusters):
    rng = np.random.default_rng(812)
    centers = rng.normal(size=(n_clusters, 3)) * 3
    labels = np.tile(np.arange(n_clusters), 4)
    points = centers[labels] + rng.normal(size=(len(labels), 3)) * 0.1
    index = DB(backend=backend)
    batch = index.get_cvi(points, labels)
    np.testing.assert_allclose(batch, direct_db(points, labels), rtol=1e-12)

    for label in (0, n_clusters):
        point = rng.normal(size=3)
        incremental = index.get_cvi(point, label)
        points = np.vstack((points, point))
        labels = np.append(labels, label)
        np.testing.assert_allclose(incremental, direct_db(points, labels), rtol=1e-12)
        np.testing.assert_array_equal(np.diag(index._R), 0)
        np.testing.assert_array_equal(index._R, index._R.T)


def test_db_does_not_evaluate_unused_diagonal():
    """Large finite scatter must not overflow just from computing self-pairs."""
    index = DB()
    index._n_clusters = 2
    index._S = [np.finfo(float).max * 0.75, 0.0]
    index._D = np.array([[0.0, 4.0], [4.0, 0.0]])
    with np.errstate(all="raise"):
        index._evaluate()
    assert index.criterion_value == index._S[0] / 4
    np.testing.assert_array_equal(np.diag(index._R), 0)
