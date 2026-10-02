"""Aggregate updates against complete-data and sequential reference paths."""

import pickle
import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.modules import _base, _batch


SUPPORTED = [cvi.CH, cvi.WB, cvi.DB, cvi.XB, cvi.GD43, cvi.GD53, cvi.PS,
             cvi.cSIL, cvi.rCIP]


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param not in request.node.callspec.params["index_type"].info.backends:
        pytest.skip("Index does not support this backend")
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


def dataset():
    rng = np.random.default_rng(136)
    labels = rng.choice([90, -7, 400], size=93)
    labels[:3] = 90
    labels[3:6] = [-7, 400, 90]
    labels[-1] = 999
    data = rng.normal(size=(len(labels), 5)) + (labels % 7)[:, None]
    return data, labels


def assert_partition(index, data, labels, *, rtol=2e-11, atol=2e-11):
    """Compare score and active sufficient/derived state with a full batch."""
    reference = type(index)()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reference.get_cvi(data, labels)
    assert index._label_map.map == reference._label_map.map
    assert index._n_samples == len(data)
    assert index._n_clusters == len(set(labels))
    assert isinstance(index._n, list)
    for key in ("_n", "_v", "_CP", "_D", "_S", "_R", "_SEP",
                "_sigma", "criterion_value"):
        if hasattr(reference, key):
            if getattr(reference, key) is None:
                assert getattr(index, key) is None
                continue
            np.testing.assert_allclose(getattr(index, key), getattr(reference, key),
                                       rtol=rtol, atol=atol, equal_nan=True,
                                       err_msg=key)
    if index._uses_compactness_stats:
        np.testing.assert_allclose(index._mu, reference._mu, rtol=rtol, atol=atol)
        assert isinstance(index._CP, list)


@pytest.mark.parametrize("index_type", SUPPORTED)
@pytest.mark.parametrize("initial", ["empty", "batch", "stream"])
def test_chunk_prefixes_match_full_partition(index_type, initial, backend):
    data, labels = dataset()
    original_data, original_labels = data.copy(), labels.copy()
    index = index_type(backend=backend)
    start = 0 if initial == "empty" else 13
    if initial == "batch":
        index.get_cvi(data[:start], labels[:start])
    elif initial == "stream":
        for sample, label in zip(data[:start], labels[:start]):
            index.get_cvi(sample, int(label))
    for size in (1, 2, 7, 31, 100):
        stop = min(start + size, len(data))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = index.update_batch(data[start:stop], labels[start:stop])
        assert isinstance(value, float)
        if len(set(labels[:stop])) > 1:
            assert_partition(index, data[:stop], labels[:stop])
        else:
            assert np.isnan(value)
        start = stop
    np.testing.assert_array_equal(data, original_data)
    np.testing.assert_array_equal(labels, original_labels)
    with pytest.raises(ValueError, match="Repeated batch"):
        index.get_cvi(data, labels)


@pytest.mark.parametrize("index_type", SUPPORTED)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_strided_input_accumulates_in_float64(index_type, dtype, backend):
    data, labels = dataset()
    data = data.astype(dtype)[::-1, ::2]
    labels = labels[::-1]
    data.setflags(write=False)
    labels.setflags(write=False)
    index = index_type(backend=backend)
    for start in range(0, len(data), 17):
        index.update_batch(data[start:start + 17], labels[start:start + 17])
    assert index._v.dtype == np.float64
    assert_partition(index, data.astype(np.float64), labels)


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_updates_continue_after_structural_operations(index_type, backend):
    data, labels = dataset()
    index = index_type(backend=backend)
    index.update_batch(data[:42], labels[:42])
    index.update_batch(data[42:], labels[42:])
    index.get_cvi(np.zeros(5), 123)
    data = np.vstack((data, np.zeros(5)))
    labels = np.r_[labels, 123]
    assert_partition(index, data, labels)

    index.remove(data[-1], 123)  # Delete the new singleton cluster.
    data, labels = data[:-1], labels[:-1]
    index.merge(90, -7)
    labels[labels == -7] = 90
    assert_partition(index, data, labels)

    subset = np.flatnonzero(labels == 90)[:4]
    centroid = data[subset].mean(axis=0)
    compactness = np.sum((data[subset] - centroid) ** 2)
    index.split(90, -20, len(subset), centroid, compactness=compactness,
                covariance=np.cov(data[subset], rowvar=False))
    labels[subset] = -20
    # Public label order follows operation history, not the regrouped data.
    new_data = np.array([[1., 2., 3., 4., 5.], [-2., -1., 0., 1., 2.]])
    index.update_batch(new_data, [90, -20])
    data, labels = np.vstack((data, new_data)), np.r_[labels, 90, -20]
    reference = index_type()
    reference.get_cvi(data, labels)
    np.testing.assert_allclose(index.criterion_value, reference.criterion_value,
                               rtol=2e-11, atol=2e-11)
    for label, slot in index._label_map.map.items():
        expected = reference._label_map.map[label]
        assert index._n[slot] == reference._n[expected]
        np.testing.assert_allclose(index._v[slot], reference._v[expected], atol=2e-11)
        if index._uses_compactness_stats:
            np.testing.assert_allclose(index._CP[slot], reference._CP[expected],
                                       rtol=2e-11, atol=2e-11)


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_empty_chunks_and_reset(index_type):
    index = index_type()
    before = pickle.dumps(index)
    assert np.isnan(index.update_batch(np.empty((0, 2)), []))
    assert pickle.dumps(index) == before
    index.update_batch([[0., 1.], [2., 3.]], [10, 20])
    before = pickle.dumps(index)
    value = index.update_batch(np.empty((0, 2)), [])
    np.testing.assert_equal(value, index.criterion_value)
    assert pickle.dumps(index) == before
    index.remove(np.array([0., 1.]), 10)
    index.remove(np.array([2., 3.]), 20)
    assert not index._is_setup
    # A fully emptied index can infer a new feature dimension.
    index.update_batch([[0., 1., 2.], [3., 4., 5.]], [90, -7])
    assert index._dim == 3


@pytest.mark.parametrize("index_type", SUPPORTED)
@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("data, labels", [
    (np.ones(2), [0]),
    (np.ones((1, 2, 1)), [0]),
    (np.ones((1, 0)), [0]),
    (np.ones((2, 2)), [100]),
    (np.ones((1, 2)), [[100]]),
    (np.ones((1, 2)), [1.5]),
    (np.ones((1, 2)), [True]),
    (np.ones((1, 2)), ["new"]),
    (np.array([[np.nan, 0]]), [100]),
    (np.array([[np.inf, 0]]), [100]),
    (np.array([[1j, 0j]]), [100]),
    (np.array([["1", "2"]]), [100]),
    (np.ones((1, 2), dtype=object), [100]),
])
def test_bad_chunk_leaves_state_unchanged(index_type, initialized, data, labels):
    index = index_type()
    if initialized:
        index.update_batch([[0., 1.], [2., 3.]], [10, 20])
    before = pickle.dumps(index)
    with pytest.raises(ValueError):
        index.update_batch(data, labels)
    assert pickle.dumps(index) == before


def test_wrong_feature_count_even_on_empty_chunk():
    index = cvi.CH()
    index.update_batch([[0., 1.]], [10])
    before = pickle.dumps(index)
    for data, labels in ((np.ones((1, 3)), [999]), (np.empty((0, 3)), [])):
        with pytest.raises(ValueError, match="Expected 2 features"):
            index.update_batch(data, labels)
        assert pickle.dumps(index) == before


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_rebuild_failure_is_atomic(index_type, monkeypatch):
    index = index_type()
    index.update_batch([[0., 1.], [2., 3.]], [10, 20])
    before = pickle.dumps(index)

    def fail_rebuild(candidate):
        candidate._v[:] = 0
        candidate._label_map.map[999] = 100
        raise ValueError("rebuild failed")

    monkeypatch.setattr(index_type, "_rebuild_after_operation", fail_rebuild)
    with pytest.raises(ValueError, match="rebuild failed"):
        index.update_batch([[9., 8.]], [999])
    assert pickle.dumps(index) == before


def test_overflow_leaves_state_unchanged():
    index = cvi.CH()
    index.update_batch([[0., 1.], [2., 3.]], [10, 20])
    before = pickle.dumps(index)
    with pytest.raises(ValueError, match="float64 range"):
        index.update_batch([[1e200, 0.], [-1e200, 0.]], [100, 100])
    assert pickle.dumps(index) == before


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
@pytest.mark.parametrize("index_type, field", [
    (cvi.PS, 1),  # Centroids, without compactness statistics.
    (cvi.CH, 2),  # Compactness.
    (cvi.CH, 3),  # Residual sums.
])
def test_nonfinite_merged_moments_leave_state_unchanged(
    index_type, field, initialized, invalid, monkeypatch,
):
    index = index_type()
    if initialized:
        index.update_batch([[0., 1.], [1., 0.], [2., 3.]], [10, 10, 20])
    before = pickle.dumps(index)
    merge = index._backend.merge_statistics

    def nonfinite_merge(*args, **kwargs):
        result = merge(*args, **kwargs)
        result[field][0] = invalid
        return result

    # A backend can return non-finite results without raising a NumPy error.
    # Check the public rejection and rollback contract at that boundary.
    with monkeypatch.context() as patch:
        patch.setattr(type(index._backend), "merge_statistics",
                      staticmethod(nonfinite_merge))
        with pytest.raises(ValueError, match="float64 range"):
            index.update_batch([[9., 8.]], [999])
    assert pickle.dumps(index) == before
    index.update_batch([[9., 8.]], [999])
    assert index._n_samples == (4 if initialized else 1)
    assert 999 in index._label_map.map


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_global_mean_leaves_state_unchanged(initialized, invalid, monkeypatch):
    index = cvi.CH()
    if initialized:
        index.update_batch([[0., 1.], [1., 0.], [2., 3.]], [10, 10, 20])
    before = pickle.dumps(index)
    with monkeypatch.context() as patch:
        patch.setattr(_batch, "centered_mean", Mock(return_value=np.full(2, invalid)))
        with pytest.raises(ValueError, match="float64 range"):
            index.update_batch([[9., 8.]], [999])
    assert pickle.dumps(index) == before
    index.update_batch([[9., 8.]], [999])
    assert index._n_samples == (4 if initialized else 1)
    np.testing.assert_array_equal(index._mu, [3., 3.] if initialized else [9., 8.])


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
@pytest.mark.parametrize("initialized", [False, True])
def test_conn_kmeans_is_unsupported(model_type, initialized):
    index = cvi.CONN(
        model_type=model_type, kmeans_k=1,
        kmeans_kwargs={"random_state": 0, "n_init": 1},
    )
    if initialized:
        index.get_cvi(np.array([[0., 0.], [1., 1.]]), np.array([10, 20]))
    before = pickle.dumps(index)
    for data, labels in (([[0., 1.]], [0]), (np.empty((0, 2)), [])):
        with pytest.raises(NotImplementedError, match="update_batch"):
            index.update_batch(data, labels)
        assert pickle.dumps(index) == before


@pytest.mark.parametrize("index_type", [cvi.CH, cvi.WB, cvi.XB])
@pytest.mark.parametrize("capacity", [None, 4])
def test_jax_is_unsupported(index_type, capacity, monkeypatch):
    monkeypatch.setattr(_base, "get_backend",
                        lambda backend: SimpleNamespace(name=backend))
    with pytest.raises(NotImplementedError, match="update_batch"):
        index_type(backend="jax", capacity=capacity).update_batch([[0., 1.]], [0])


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_large_offset_centered_moments(index_type, backend):
    rng = np.random.default_rng(817)
    data = 1e12 + rng.normal(size=(181, 4))
    labels = np.arange(len(data)) % 3
    index = index_type(backend=backend)
    for start in range(0, len(data), 13):
        index.update_batch(data[start:start + 13], labels[start:start + 13])
    for label, slot in index._label_map.map.items():
        # Long double oracle measures moments about the actual stored center.
        centered = data[labels == label].astype(np.longdouble) - index._v[slot]
        if index._uses_compactness_stats:
            np.testing.assert_allclose(index._CP[slot], np.sum(centered ** 2),
                                       rtol=2e-12, atol=2e-12)
            np.testing.assert_allclose(index._G[slot], np.sum(centered, axis=0),
                                       rtol=2e-12, atol=2e-12)
    assert np.isfinite(index.criterion_value)


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_degenerate_partition_matches_existing_definition(index_type, backend):
    for data in (np.ones((6, 2)), np.array([[-1., 2.]] * 3 + [[1., -2.]] * 3)):
        labels = np.array([90, -7, 400, 90, -7, 400])
        index = index_type(backend=backend)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            index.update_batch(data[:2], labels[:2])
            index.update_batch(data[2:], labels[2:])
        assert_partition(index, data, labels)


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_aggregate_path_does_not_scan_samples(index_type, monkeypatch):
    scan = Mock(side_effect=AssertionError("aggregate updates must not call _param_inc"))
    monkeypatch.setattr(index_type, "_param_inc", scan)
    data, labels = dataset()
    index = index_type()
    index.update_batch(data[:37], labels[:37])
    index.update_batch(data[37:], labels[37:])
    scan.assert_not_called()
    assert_partition(index, data, labels)


def test_unsigned_labels_keep_full_range_and_first_seen_order():
    labels = np.array([2**64 - 1, 2**63, 2**64 - 1], dtype=np.uint64)
    index = cvi.CH()
    index.update_batch([[0.], [2.], [1.]], labels)
    index.update_batch([[3.], [4.]], np.array([2**63, 1], dtype=np.uint64))
    assert index._label_map.map == {2**64 - 1: 0, 2**63: 1, 1: 2}
