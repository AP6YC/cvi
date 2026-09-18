"""Regression oracles for shared statistics and centroid-distance kernels."""

import warnings

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.backends import get_backend


CHANGED_CVIS = [cvi.CH, cvi.WB, cvi.DB, cvi.XB, cvi.GD43, cvi.GD53,
                cvi.PS, cvi.cSIL]
DISTANCE_CVIS = [cvi.DB, cvi.XB, cvi.GD43, cvi.PS]


def reference_batch(cvi_type, data, labels):
    """Build state using the pre-refactor batch definitions, without kernels.

    Deliberately scan labels and calculate cSIL distances from samples. This
    independently checks grouping, raw/centered moments, and matrix orientation.
    Evaluators are unchanged by the refactor and run on the reference state.
    """
    index = cvi_type()
    index._setup_batch(data)
    unique = index._setup_batch_labels(labels)
    index._n_clusters = k = len(unique)
    groups = [data[labels == label, :] for label in unique]
    index._n = [len(group) for group in groups]
    index._v = np.zeros((k, data.shape[1]))
    for i, group in enumerate(groups):
        index._v[i] = group.mean(axis=0)

    if cvi_type is not cvi.cSIL:
        index._mu = data.mean(axis=0)
    if cvi_type is not cvi.PS:
        index._CP = [np.sum((group - index._v[i]) ** 2)
                     for i, group in enumerate(groups)]
        index._G = np.zeros_like(index._v)

    if cvi_type in (cvi.CH, cvi.WB):
        index._SEP = np.array([
            len(group) * np.sum((index._v[i] - index._mu) ** 2)
            for i, group in enumerate(groups)
        ])
    elif cvi_type is cvi.cSIL:
        index._CP = [np.sum(group ** 2) for group in groups]
        index._G = np.asarray([group.sum(axis=0) for group in groups],
                              dtype=float)
        index._S = np.zeros((k, k))
        for i, group in enumerate(groups):
            for j in range(k):
                # Use the original sum-of-per-sample-distances definition.
                index._S[i, j] = sum(
                    np.sum((group - index._v[j]) ** 2, axis=1)
                ) / len(group)
    else:
        index._D = np.zeros((k, k))
        for i in range(k - 1):
            for j in range(i + 1, k):
                if cvi_type is cvi.GD53:
                    value = ((index._CP[i] + index._CP[j])
                             / (index._n[i] + index._n[j]))
                else:
                    value = np.sum((index._v[i] - index._v[j]) ** 2)
                    if cvi_type is cvi.GD43:
                        value = np.sqrt(value)
                index._D[i, j] = index._D[j, i] = value
        if cvi_type is cvi.DB:
            index._S = [cp / n for cp, n in zip(index._CP, index._n)]
    index._evaluate()
    return index


def assert_state_equal(actual, expected):
    assert actual._label_map.map == expected._label_map.map
    assert actual._n_samples == expected._n_samples
    assert actual._n_clusters == expected._n_clusters
    assert actual._is_setup == expected._is_setup
    assert isinstance(actual._n, list)
    assert isinstance(actual._CP, list)
    for name in ("_n", "_v", "_CP", "_G", "_D", "_S", "_SEP", "_mu",
                 "_R", "_sil_coefs", "_PS_i", "criterion_value"):
        if hasattr(expected, name):
            np.testing.assert_allclose(
                getattr(actual, name), getattr(expected, name),
                rtol=2e-12, atol=2e-12, equal_nan=True, err_msg=name,
            )


def batch_case(case, dtype):
    rng = np.random.default_rng(318)
    labels = rng.choice([90, -7, 400], size=83)
    labels[:3] = [90, -7, 400]
    labels[-1] = 12345  # A singleton, in addition to unbalanced clusters.
    data = rng.normal(size=(len(labels), 6)) + (labels % 7)[:, None]
    if case == "offset":
        data += 1e5 if dtype == np.float32 else 1e12
    elif case == "coincident":
        labels = np.array([90, -7, 400, 90, -7, 400])
        data = np.array([[-1., 2.]] * 3 + [[1., -2.]] * 3)
    elif case == "identical":
        data[:] = 1
    elif case == "one_feature":
        data = data[:, :1]
    data = data.astype(dtype)
    if case == "strided":
        data = data[::-1, ::2]
        labels = labels[::-1]
        assert not data.flags.c_contiguous
    return data, labels


@pytest.mark.parametrize("cvi_type", CHANGED_CVIS)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("case", [
    "random", "offset", "strided", "coincident", "identical", "one_feature",
])
def test_batch_matches_direct_sample_definitions(cvi_type, dtype, case, backend):
    data, labels = batch_case(case, dtype)
    original_data, original_labels = data.copy(), labels.copy()
    # Undefined scores (e.g. coincident centroids) are NaN and warn on batches.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        expected = reference_batch(cvi_type, data, labels)
    actual = cvi_type(backend=backend)
    if np.isnan(expected.criterion_value):
        with pytest.warns(
            RuntimeWarning,
            match=f"{cvi_type.__name__} is undefined for the supplied batch",
        ):
            actual.get_cvi(data, labels)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual.get_cvi(data, labels)
    assert_state_equal(actual, expected)
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(labels, original_labels)


@pytest.mark.parametrize("cvi_type", CHANGED_CVIS)
def test_reference_batch_state_remains_equivalent_through_operations(
    cvi_type, backend,
):
    data, labels = batch_case("random", np.float64)
    actual = cvi_type(backend=backend)
    actual.get_cvi(data, labels)
    expected = reference_batch(cvi_type, data, labels)
    for index in (actual, expected):
        index.get_cvi(np.arange(data.shape[1]) * 0.1, 90)
    assert_state_equal(actual, expected)
    for index in (actual, expected):
        index.get_cvi(np.arange(data.shape[1]) * 0.2, 999)
    assert_state_equal(actual, expected)
    for index in (actual, expected):
        index.remove(data[-1], int(labels[-1]))
    assert_state_equal(actual, expected)
    for index in (actual, expected):
        index.merge(-7, 90)
    assert_state_equal(actual, expected)


@pytest.mark.parametrize("cvi_type", DISTANCE_CVIS)
def test_distance_state_after_streaming_and_structural_operations(cvi_type, backend):
    data, labels = batch_case("random", np.float64)
    actual = cvi_type(backend=backend)

    def check_distances():
        expected = np.array([
            [np.sum((a - b) ** 2) for b in actual._v] for a in actual._v
        ])
        if cvi_type is cvi.GD43:
            expected = np.sqrt(expected)
        np.testing.assert_allclose(actual._D, expected, rtol=2e-15, atol=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        for sample, label in zip(data, labels):
            actual.get_cvi(sample, int(label))
            check_distances()
        actual.remove(data[-1], int(labels[-1]))
        check_distances()
        actual.merge(-7, 90)
        check_distances()


@pytest.mark.parametrize("n_clusters", [0, 1, 2, 17])
@pytest.mark.parametrize("squared", [True, False])
def test_pairwise_kernel_empty_singleton_and_large_offsets(
    n_clusters, squared, backend,
):
    centers = np.random.default_rng(19).normal(size=(n_clusters, 5)) + 1e12
    expected = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            expected[i, j] = np.sum((centers[i] - centers[j]) ** 2)
    if not squared:
        expected = np.sqrt(expected)
    actual = get_backend(backend).pairwise_centroid_distances(centers, squared=squared)
    np.testing.assert_array_equal(actual, expected)


def test_grouping_retains_sample_order_and_first_seen_cluster_order(backend):
    data, labels = batch_case("random", np.float64)
    index = cvi.XB(backend=backend)
    order, offsets = index._setup_batch_statistics(data, labels)
    assert list(index._label_map.map) == list(dict.fromkeys(labels))
    for label, cluster in index._label_map.map.items():
        np.testing.assert_array_equal(
            order[offsets[cluster]:offsets[cluster + 1]],
            np.flatnonzero(labels == label),
        )
