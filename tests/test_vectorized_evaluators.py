"""Independent state checks for vectorized shared CVI calculations."""

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.backends import get_backend


def stream_case(n_clusters=10):
    rng = np.random.default_rng(524)
    labels = np.tile(np.arange(n_clusters), 4)
    centers = rng.normal(size=(n_clusters, 3)) * 3
    points = centers[labels] + rng.normal(size=(len(labels), 3)) * 0.2
    return points, labels


def direct_gd53_matrix(index):
    compactness = np.asarray(index._CP)
    counts = np.asarray(index._n)
    matrix = ((compactness[:, None] + compactness[None, :])
              / (counts[:, None] + counts[None, :]))
    np.fill_diagonal(matrix, 0.0)
    return matrix


@pytest.mark.parametrize("n_clusters", [2, 10, 128])
def test_gd53_dispersion_state_matches_direct_definition(
    backend, n_clusters,
):
    points, labels = stream_case(n_clusters)
    index = cvi.GD53(backend=backend)
    index.get_cvi(points, labels)
    np.testing.assert_allclose(index._D, direct_gd53_matrix(index), rtol=2e-15)

    for label in (0, n_clusters):
        value = index.get_cvi(np.arange(points.shape[1]) * 0.1, label)
        expected = direct_gd53_matrix(index)
        np.testing.assert_allclose(index._D, expected, rtol=2e-15)
        off_diagonal = expected[np.triu_indices(index._n_clusters, k=1)]
        intra = 2 * np.max(np.asarray(index._CP) / np.asarray(index._n))
        np.testing.assert_allclose(value, np.min(off_diagonal) / intra,
                                   rtol=2e-15)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_minimum_off_diagonal_restores_matrix(backend):
    if backend == "numba":
        pytest.importorskip("numba")
    matrix = np.array([
        [9.0, 4.0, 0.0],
        [4.0, 8.0, 3.0],
        [0.0, 3.0, 7.0],
    ])
    original = matrix.copy()
    assert get_backend(backend).minimum_off_diagonal(matrix) == 0.0
    np.testing.assert_array_equal(matrix, original)
