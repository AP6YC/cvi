"""Fuzzy CONN chunks preserve the complete incremental learning trajectory."""

from copy import deepcopy
import pickle
import warnings

import numpy as np
import pytest

import src.cvi as cvi


def dataset():
    rng = np.random.default_rng(136)
    data = rng.uniform(0.05, 0.95, size=(32, 2))
    data[:4] = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
    labels = rng.choice([90, -7, 400], size=len(data))
    labels[:4] = [90, 90, -7, -7]
    labels[-1] = 999
    return data, labels


def assert_same_state(actual, expected):
    for name in ("_n_samples", "_n_clusters", "_dim", "_is_setup",
                 "_rev_map", "_intra_conn", "_inter_conn", "criterion_value"):
        np.testing.assert_equal(getattr(actual, name), getattr(expected, name))
    assert actual._label_map.map == expected._label_map.map
    for name in ("_CADJ", "_CONN", "_INTRA", "_INTER", "_cluster_cardinality"):
        np.testing.assert_array_equal(getattr(actual, name).array,
                                      getattr(expected, name).array)
    assert actual._artmap.map == expected._artmap.map
    np.testing.assert_array_equal(actual._artmap.labels_, expected._artmap.labels_)
    a, b = actual._artmap.module_a, expected._artmap.module_a
    assert a.params == b.params
    assert a.sample_counter_ == b.sample_counter_
    np.testing.assert_array_equal(a.weight_sample_counter_, b.weight_sample_counter_)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    np.testing.assert_array_equal(a.W, b.W)


@pytest.mark.parametrize("initial", ["empty", "batch", "stream"])
@pytest.mark.parametrize("sizes", [(1, 1, 3, 7, 40), (2, 5, 50), (50,)])
@pytest.mark.parametrize("rho,beta", [(0.9, 1.0), (0.6, 0.4)])
def test_chunk_boundaries_preserve_incremental_state(initial, sizes, rho, beta):
    data, labels = dataset()
    data.setflags(write=False)
    labels.setflags(write=False)
    index = cvi.CONN(model_type="Fuzzy", rho=rho, beta=beta)
    reference = cvi.CONN(model_type="Fuzzy", rho=rho, beta=beta)
    start = 0 if initial == "empty" else 8
    if initial == "batch":
        # These first eight samples already span [0, 1] in each feature.
        index.get_cvi(data[:start], labels[:start])
    else:
        for sample, label in zip(data[:start], labels[:start]):
            index.get_cvi(sample, int(label))
    for sample, label in zip(data[:start], labels[:start]):
        reference.get_cvi(sample, int(label))
    for size in sizes:
        stop = min(start + size, len(data))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            value = index.update_batch(data[start:stop], labels[start:stop])
        for sample, label in zip(data[start:stop], labels[start:stop]):
            reference.get_cvi(sample, int(label))
        assert isinstance(value, float)
        assert_same_state(index, reference)
        start = stop
    # Ordinary incremental updates can continue after any chunk boundary.
    index.get_cvi(np.array([0.3, 0.4]), 123)
    reference.get_cvi(np.array([0.3, 0.4]), 123)
    assert_same_state(index, reference)


def test_chunks_continue_after_prototype_operations():
    data, labels = dataset()
    index = cvi.CONN(model_type="Fuzzy")
    index.update_batch(data[:4], labels[:4])
    index.split(90, 50, [1])
    index.merge(-7, 50)
    reference = deepcopy(index)
    index.update_batch(data[4:], labels[4:])
    for sample, label in zip(data[4:], labels[4:]):
        reference.get_cvi(sample, int(label))
    assert_same_state(index, reference)


@pytest.mark.parametrize("initialized", [False, True])
def test_empty_chunk_is_noop(initialized):
    index = cvi.CONN(model_type="Fuzzy")
    if initialized:
        data, labels = dataset()
        index.update_batch(data, labels)
    before = pickle.dumps(index)
    value = index.update_batch(np.empty((0, 2)), [])
    np.testing.assert_equal(value, index.criterion_value)
    assert pickle.dumps(index) == before


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("data,labels", [
    (np.ones(2), [0]),
    (np.ones((1, 0)), [0]),
    (np.ones((2, 2)), [0]),
    (np.ones((1, 2)), [[0]]),
    (np.ones((1, 2)), [1.5]),
    (np.ones((1, 2)), [True]),
    (np.ones((1, 2), dtype=complex), [0]),
    ([[0., 0.], [np.nan, 0.]], [0, 1]),
    ([[0., 0.], [np.inf, 0.]], [0, 1]),
    ([[0., 0.], [-0.1, 0.]], [0, 1]),
    ([[0., 0.], [1.1, 0.]], [0, 1]),
])
def test_invalid_chunk_leaves_state_unchanged(initialized, data, labels):
    index = cvi.CONN(model_type="Fuzzy")
    if initialized:
        index.get_cvi(np.array([0., 1.]), 90)
    before = pickle.dumps(index)
    with pytest.raises(ValueError):
        index.update_batch(data, labels)
    assert pickle.dumps(index) == before


@pytest.mark.parametrize("data,labels", [(np.ones((1, 3)), [1]),
                                        (np.empty((0, 3)), [])])
def test_feature_count_cannot_change(data, labels):
    index = cvi.CONN(model_type="Fuzzy")
    index.get_cvi(np.array([0., 1.]), 90)
    before = pickle.dumps(index)
    with pytest.raises(ValueError, match="Expected 2 features"):
        index.update_batch(data, labels)
    assert pickle.dumps(index) == before


@pytest.mark.parametrize("initialized", [False, True])
def test_learning_failure_is_atomic(initialized, monkeypatch):
    data, labels = dataset()
    index = cvi.CONN(model_type="Fuzzy")
    if initialized:
        index.update_batch(data[:4], labels[:4])
    before = pickle.dumps(index)
    original = cvi.CONN._param_inc

    def fail_after_learning(candidate, sample, label):
        original(candidate, sample, label)
        if label == 999:
            raise RuntimeError("learning failed")

    monkeypatch.setattr(cvi.CONN, "_param_inc", fail_after_learning)
    with pytest.raises(RuntimeError, match="learning failed"):
        index.update_batch(data, labels)
    assert pickle.dumps(index) == before


def test_strided_float32_data_and_large_integer_labels():
    data, _ = dataset()
    data = data.astype(np.float32)[::2, ::-1]
    labels = np.array([2**63, 2**64 - 1] * 8, dtype=np.uint64)
    index = cvi.CONN(model_type="Fuzzy", check_incremental_normalized=False)
    reference = cvi.CONN(model_type="Fuzzy", check_incremental_normalized=False)
    index.update_batch(data, labels)
    for sample, label in zip(data, labels):
        reference.get_cvi(sample, int(label))
    assert_same_state(index, reference)
