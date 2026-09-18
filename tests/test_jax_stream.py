"""Fixed-capacity JAX additions preserve incremental statistics and failures."""

import copy
import pickle
from functools import partial

import numpy as np
import pytest

import src.cvi as cvi


INDICES = [cvi.CH, cvi.WB, cvi.XB]


def samples(case="random", dtype=np.float64):
    rng = np.random.default_rng(53)
    data = rng.normal(size=(30, 3))
    labels = np.array([90, 90, -7, -7, 500, 90] * 5)
    data += (labels == 500)[:, None] * 4
    if case == "offset":
        data += 1e6
    elif case == "identical":
        data[:] = 2
    elif case == "one_cluster":
        labels[:] = 90
    return data.astype(dtype), labels


def assert_stream_matches(actual, expected, atol=1e-9):
    state = actual.stream_state
    k = expected._n_clusters
    assert actual._label_map.map == expected._label_map.map
    assert actual._n_samples == expected._n_samples
    np.testing.assert_array_equal(np.asarray(state.counts)[:k], expected._n)
    np.testing.assert_array_equal(state.active, np.arange(actual.capacity) < k)
    for left, right in [(state.centroids[:k], expected._v),
                        (state.compactness[:k], expected._CP),
                        (state.residuals[:k], expected._G),
                        (state.mean, expected._mu),
                        (actual.criterion_value, expected.criterion_value)]:
        np.testing.assert_allclose(left, right, rtol=2e-8, atol=atol, equal_nan=True)
    assert np.all(np.asarray(state.counts)[k:] == 0)
    if isinstance(expected, cvi.XB):
        np.testing.assert_allclose(state.distances[:k, :k], expected._D,
                                   rtol=2e-8, atol=atol)


@pytest.mark.parametrize("index_type", INDICES)
@pytest.mark.parametrize("case", ["random", "offset", "identical", "one_cluster"])
def test_sample_stream_preserves_recurrence(jax_runtime, index_type, case):
    data, labels = samples(case)
    actual = index_type(backend="jax", capacity=7)
    expected = index_type()
    with np.errstate(divide="ignore", invalid="ignore"):
        for sample, label in zip(data, labels):
            value = actual.get_cvi(sample, int(label))
            expected.get_cvi(sample, int(label))
            assert isinstance(value, float)
            assert_stream_matches(actual, expected, atol=1e-8)
    assert actual.stream_state.centroids.shape == (7, 3)
    assert all(isinstance(a, jax_runtime[0].Array) for a in actual.stream_state)


@pytest.mark.parametrize("index_type", INDICES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("return_history", [True, False])
def test_chunks_and_batch_to_stream(jax_runtime, index_type, dtype, return_history):
    data, labels = samples(dtype=dtype)
    actual = index_type(backend="jax", capacity=5)
    expected = index_type()
    actual.get_cvi(data[:4], labels[:4])
    expected.get_cvi(data[:4], labels[:4])
    # Stream arithmetic intentionally uses float64, even for float32 input.
    for start, stop in [(4, 11), (11, 30)]:
        history = [expected.get_cvi(x.astype(np.float64), int(y))
                   for x, y in zip(data[start:stop], labels[start:stop])]
        result = actual.update_many(data[start:stop], labels[start:stop],
                                    return_history=return_history)
        np.testing.assert_allclose(result, history if return_history else history[-1],
                                   rtol=2e-8, atol=1e-9, equal_nan=True)
        assert_stream_matches(actual, expected)
    with pytest.raises(ValueError, match="Repeated batch"):
        actual.get_cvi(data, labels)
    restored = pickle.loads(pickle.dumps(actual))
    restored.get_cvi(data[0].astype(np.float64), int(labels[0]))
    expected.get_cvi(data[0].astype(np.float64), int(labels[0]))
    assert_stream_matches(restored, expected)


@pytest.mark.parametrize("index_type", INDICES)
def test_capacity_and_chunk_errors_are_atomic(jax_runtime, index_type):
    data, labels = samples()
    index = index_type(backend="jax", capacity=2)
    with pytest.raises(ValueError, match="capacity"):
        index.get_cvi(data, labels)
    assert index.stream_state is None and index._label_map.map == {}
    index.update_many(data[:4], labels[:4])
    before = copy.deepcopy(index)
    actions = [
        lambda: index.get_cvi(data[4], int(labels[4])),
        lambda: index.update_many(data[3:6], labels[3:6]),
        lambda: index.update_many(np.ones((2, 4)), labels[:2]),
        lambda: index.update_many(data[:2], labels[:1]),
        lambda: index.update_many(data[:2], np.array([90., 90.])),
        lambda: index.update_many(np.array([[np.nan, 0, 0]]), labels[:1]),
        lambda: index.update_many(np.array([[np.inf, 0, 0]]), labels[:1]),
        lambda: index.get_cvi(data[0], True),
        lambda: index.get_cvi(data[0], [90]),
        lambda: index.get_cvi(data[0].astype(complex), 90),
        lambda: index.update_many(data[:2], labels[:2], return_history=1),
    ]
    for action in actions:
        with pytest.raises(ValueError):
            action()
        assert index._label_map.map == before._label_map.map
        assert index._n_samples == before._n_samples
        for a, b in zip(index.stream_state, before.stream_state):
            np.testing.assert_array_equal(a, b)
    for action in [lambda: index.remove(data[0], 90), lambda: index.merge(90, -7)]:
        with pytest.raises(NotImplementedError, match="remove or merge"):
            action()
    # Filling capacity never prevents additions to existing clusters.
    index.get_cvi(data[0], 90)
    assert index._n_samples == 5


@pytest.mark.parametrize("capacity", [0, -1, True, 2.5, "3"])
def test_invalid_capacity(jax_runtime, capacity):
    with pytest.raises(ValueError, match="capacity"):
        cvi.CH(backend="jax", capacity=capacity)


def test_capacity_opt_in_and_empty_chunks(jax_runtime):
    for backend in ["numpy", "numba"]:
        with pytest.raises(ValueError, match="only"):
            cvi.CH(backend=backend, capacity=2)
    with pytest.raises(NotImplementedError, match="capacity"):
        cvi.CH().update_many(np.ones((1, 2)), [0])
    obj = cvi.CH(backend="jax", capacity=1)
    with pytest.raises(AttributeError):
        obj.capacity = 2
    with pytest.raises(AttributeError):
        obj.stream_state = None
    assert obj.update_many(np.empty((0, 2)), np.array([], dtype=int)).size == 0
    assert obj.stream_state is None
    assert obj.get_cvi(np.array([1., 2.]), 77) == 0
    assert obj.update_many(np.empty((0, 2)), np.array([], dtype=int),
                           return_history=False) == 0
    assert obj.get_cvi(np.array([2., 3.]), 77) == 0


@pytest.mark.parametrize("index_type", INDICES)
def test_functional_scan_jit_sparse_slots_and_masks(jax_runtime, index_type):
    jax, f = jax_runtime
    name = index_type.info.name_short
    data, labels = samples()
    slots = np.array([{90: 4, -7: 1, 500: 6}[int(y)] for y in labels])
    state = f.empty_stream(capacity=8, n_features=3, index=name)
    traces = []

    @jax.jit
    def update(state, x, slot):
        traces.append(True)
        return f.stream_update(state, x, slot, index=name)

    sequential = state
    values = []
    for x, slot in zip(data, slots):
        sequential, score = update(sequential, x, slot)
        values.append(float(score))
    assert len(traces) == 1  # new clusters do not change traced shapes
    assert sequential.distances.shape == ((8, 8) if name == "XB" else (0, 0))
    expected = index_type()
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = [expected.get_cvi(x, int(y)) for x, y in zip(data, labels)]
    np.testing.assert_allclose(values, reference, rtol=1e-10, equal_nan=True)
    for history in [True, False]:
        scan = jax.jit(partial(f.stream_chunk, index=name, return_history=history))
        result, output = scan(state, data, slots)
        np.testing.assert_allclose(output, values if history else values[-1],
                                   rtol=1e-10, equal_nan=True)
        for a, b in zip(result, sequential):
            np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(f.evaluate_stream(result, index=name), values[-1])
    np.testing.assert_array_equal(
        result.active, [False, True, False, False, True, False, True, False],
    )


@pytest.mark.parametrize("index", ["CH", "WB", "XB"])
def test_functional_invalid_chunk_is_atomic_under_jit(jax_runtime, index):
    jax, f = jax_runtime
    state = f.empty_stream(capacity=2, n_features=1, index=index)
    scan = jax.jit(partial(f.stream_chunk, index=index))
    for data, slots in [(np.ones((2, 1)), [0, 2]),
                        (np.ones((2, 1)), [0, -1]),
                        (np.array([[1.], [np.nan]]), [0, 1])]:
        result, scores = scan(state, data, np.array(slots))
        assert np.all(np.isnan(scores))
        for a, b in zip(result, state):
            np.testing.assert_array_equal(a, b)
    result, score = f.stream_update(state, np.array([1.]), 2, index=index)
    assert np.isnan(score)
    np.testing.assert_array_equal(result.counts, state.counts)
    result, history = scan(state, np.empty((0, 1)), np.array([], dtype=int))
    assert history.shape == (0,)
    assert f.evaluate_stream(result, index=index) == 0


@pytest.mark.parametrize("index", ["CH", "WB", "XB"])
def test_functional_batch_transition(jax_runtime, index):
    jax, f = jax_runtime
    data, labels = samples()
    slots = np.array([{90: 0, -7: 1, 500: 2}[int(y)] for y in labels])
    batch = f.batch_state(data[:4], slots[:4], n_clusters=2)
    with pytest.raises(ValueError, match="capacity"):
        f.stream_from_batch(batch, capacity=1, index=index)
    state = jax.jit(partial(f.stream_from_batch, capacity=5, index=index))(batch)
    np.testing.assert_allclose(f.evaluate_stream(state, index=index),
                               f.evaluate(batch, index=index))
    state, value = f.stream_chunk(
        state, data[4:], slots[4:], index=index, return_history=False,
    )
    expected = getattr(cvi, index)(backend="jax", capacity=5)
    expected.get_cvi(data[:4], labels[:4])
    expected.update_many(data[4:], labels[4:])
    np.testing.assert_allclose(value, expected.criterion_value)
    np.testing.assert_allclose(state.compactness, expected.stream_state.compactness)


def test_functional_validation_and_x64(jax_runtime):
    jax, f = jax_runtime
    state = f.empty_stream(capacity=2, n_features=2, index="CH")
    for data, slots in [(np.ones((2, 3)), np.array([0, 1])),
                        (np.ones((2, 2)), np.array([0])),
                        (np.ones((2, 2)), np.array([0., 1.])),
                        (np.ones((2, 2), dtype=complex), np.array([0, 1]))]:
        with pytest.raises(ValueError):
            f.stream_chunk(state, data, slots, index="CH")
    with pytest.raises(ValueError, match="match"):
        f.evaluate_stream(state, index="XB")
    for size in [0, -1, True, 2.5]:
        with pytest.raises(ValueError):
            f.empty_stream(capacity=size, n_features=2, index="CH")
        with pytest.raises(ValueError):
            f.empty_stream(capacity=2, n_features=size, index="CH")
    obj = cvi.CH(backend="jax", capacity=2)
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="64-bit"):
        obj.get_cvi(np.ones(2), 0)
    with pytest.raises(ValueError, match="64-bit"):
        f.stream_update(state, np.ones(2), 0, index="CH")
    assert obj.stream_state is None


@pytest.mark.parametrize("index_type", INDICES)
def test_large_finite_new_cluster_has_zero_compactness(jax_runtime, index_type):
    actual = index_type(backend="jax", capacity=3)
    assert actual.get_cvi(np.array([1e200, -1e200]), 11) == 0
    np.testing.assert_array_equal(actual.stream_state.compactness, 0.0)
    np.testing.assert_array_equal(actual.stream_state.residuals, 0.0)
