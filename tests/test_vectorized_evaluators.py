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


@pytest.mark.parametrize("cvi_type", [cvi.CH, cvi.WB])
def test_separation_state_matches_direct_definition(backend, cvi_type):
    points, labels = stream_case()
    index = cvi_type(backend=backend)
    for point, label in zip(points, labels):
        index.get_cvi(point, int(label))
        expected = np.asarray(index._n) * np.sum(
            (index._v - index._mu) ** 2, axis=1,
        )
        np.testing.assert_allclose(index._SEP, expected, rtol=2e-15, atol=0)


def direct_ps(index):
    centroid_mean = np.mean(index._v, axis=0)
    beta = np.mean(np.sum((index._v - centroid_mean) ** 2, axis=1))
    pairwise = np.sum(
        (index._v[:, None, :] - index._v[None, :, :]) ** 2, axis=2,
    )
    np.fill_diagonal(pairwise, np.inf)
    scores = (
        np.asarray(index._n) / np.max(index._n)
        - np.exp(-np.min(pairwise, axis=1) / beta)
    )
    return centroid_mean, beta, scores


def test_ps_shared_evaluator_matches_direct_definition(backend):
    points, labels = stream_case()
    index = cvi.PS(backend=backend)
    for point, label in zip(points, labels):
        value = index.get_cvi(point, int(label))
        if index._n_clusters < 2 or index._beta_t == 0:
            continue
        centroid_mean, beta, scores = direct_ps(index)
        np.testing.assert_allclose(index._v_bar, centroid_mean, rtol=2e-15)
        np.testing.assert_allclose(index._beta_t, beta, rtol=2e-15)
        np.testing.assert_allclose(index._PS_i, scores, rtol=2e-15)
        np.testing.assert_allclose(value, np.sum(scores), rtol=2e-15)


def test_ps_batch_and_incremental_state_layout_match(backend):
    """PS retains the common empty fields in both initialization modes."""
    points, labels = stream_case()
    batch = cvi.PS(backend=backend)
    incremental = cvi.PS(backend=backend)

    batch_value = batch.get_cvi(points, labels)
    for point, label in zip(points, labels):
        incremental_value = incremental.get_cvi(point, int(label))

    for index in (batch, incremental):
        assert index._CP == []
        assert index._G.shape == (0, points.shape[1])

    assert batch._label_map.map == incremental._label_map.map
    np.testing.assert_array_equal(batch._n, incremental._n)
    np.testing.assert_allclose(batch._v, incremental._v, rtol=2e-15)
    np.testing.assert_allclose(batch._D, incremental._D, rtol=2e-15)
    np.testing.assert_allclose(batch._PS_i, incremental._PS_i, rtol=2e-15)
    np.testing.assert_allclose(batch_value, incremental_value, rtol=2e-15)


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
