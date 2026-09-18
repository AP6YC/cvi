"""JAX batch compatibility and transformation tests (optional dependency)."""

import copy
import importlib
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import src.cvi as cvi
from test_kernels import assert_state_equal, batch_case, reference_batch


INDICES = [cvi.CH, cvi.WB, cvi.XB]


@pytest.fixture
def jax_runtime():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax, importlib.import_module("src.cvi.jax")
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("index_type", INDICES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
@pytest.mark.parametrize("case", [
    "random", "offset", "strided", "coincident", "identical", "one_feature",
])
def test_jax_object_matches_reference_state(jax_runtime, index_type, dtype, case):
    data, labels = batch_case(case, dtype)
    before_data, before_labels = data.copy(), labels.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = reference_batch(index_type, data, labels)
        actual = index_type(backend="jax")
        value = actual.get_cvi(data, labels)
    assert isinstance(value, float)
    assert actual.backend == "jax"
    assert_state_equal(actual, expected)
    np.testing.assert_array_equal(data, before_data)
    np.testing.assert_array_equal(labels, before_labels)


@pytest.mark.parametrize("index_type", INDICES)
def test_jax_batch_only_errors_are_atomic(jax_runtime, index_type):
    data, labels = batch_case("random", np.float64)
    index = index_type(backend="jax")
    with pytest.raises(NotImplementedError, match="batch only"):
        index.get_cvi(data[0], int(labels[0]))
    assert not index._is_setup and index._label_map.map == {}
    index.get_cvi(data, labels)
    before = copy.deepcopy(index)
    for action in (lambda: index.get_cvi(data[0], int(labels[0])),
                   lambda: index.remove(data[0], int(labels[0])),
                   lambda: index.merge(int(labels[0]), int(labels[1]))):
        with pytest.raises(NotImplementedError, match="batch only"):
            action()
        assert_state_equal(index, before)
    with pytest.raises(ValueError, match="Repeated batch"):
        index.get_cvi(data, labels)
    assert_state_equal(index, before)
    restored = pickle.loads(pickle.dumps(index))
    assert restored.backend == "jax"
    assert_state_equal(restored, before)


@pytest.mark.parametrize("data,labels", [
    (np.zeros((0, 2)), np.array([], dtype=int)),
    (np.zeros((3, 0)), np.array([0, 1, 1])),
    (np.zeros((3, 2, 1)), np.array([0, 1, 1])),
    (np.zeros((3, 2)), np.array([0, 1])),
    (np.zeros((3, 2)), np.array([[0], [1], [1]])),
    (np.zeros((3, 2)), np.array([0., 1., 1.])),
    (np.zeros((3, 2)), np.array([0, 0, 0])),
    (np.zeros((3, 2), dtype=complex), np.array([0, 1, 1])),
])
def test_jax_invalid_batches_do_not_initialize(jax_runtime, data, labels):
    index = cvi.XB(backend="jax")
    with pytest.raises(ValueError):
        index.get_cvi(data, labels)
    assert index._n_samples == 0
    assert index._label_map.map == {}
    assert not index._is_setup


def test_jax_requires_x64_without_changing_configuration(jax_runtime):
    jax, functional = jax_runtime
    jax.config.update("jax_enable_x64", False)
    with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
        cvi.XB(backend="jax")
    with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
        functional.batch_state(np.ones((3, 2)), np.array([0, 1, 1]), n_clusters=2)
    assert not jax.config.x64_enabled


@pytest.mark.parametrize("index_type", [
    cvi.DB, cvi.GD43, cvi.GD53, cvi.PS, cvi.cSIL, cvi.rCIP, cvi.CONN,
])
def test_unsupported_jax_indices_are_explicit(index_type):
    with pytest.raises(NotImplementedError, match="does not support the jax"):
        index_type(backend="jax")


def test_numpy_imports_without_jax():
    code = '''
import importlib.abc
import sys
class NoJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ("jax", "jaxlib") or fullname.startswith(("jax.", "jaxlib.")):
            raise ModuleNotFoundError("JAX blocked for test", name="jax")
sys.meta_path.insert(0, NoJax())
import numpy as np
import src.cvi as cvi
cvi.XB().get_cvi(np.array([[0.], [1.], [4.], [5.]]), np.array([0, 0, 1, 1]))
assert "jax" not in sys.modules
assert "src.cvi.jax" not in sys.modules
try:
    cvi.XB(backend="jax")
except ImportError as error:
    assert 'pip install "cvi[jax]"' in str(error)
else:
    raise AssertionError("Missing JAX must not silently select NumPy")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("index_type", INDICES)
def test_jax_functional_jit_vmap_and_grad(jax_runtime, index_type):
    jax, functional = jax_runtime
    data, labels = batch_case("random", np.float64)
    unique = list(dict.fromkeys(labels))
    dense = np.array([unique.index(label) for label in labels])

    def score(x):
        return functional.batch_cvi(
            x, dense, n_clusters=len(unique), index=index_type.__name__,
        )
    compiled = jax.jit(score)
    value = compiled(jax.numpy.asarray(data))
    assert isinstance(value, jax.Array)
    np.testing.assert_allclose(value, index_type().get_cvi(data, labels), rtol=2e-12)
    batches = np.stack((data, data * 2 + 7, data + 3))
    expected = [index_type().get_cvi(x, labels) for x in batches]
    np.testing.assert_allclose(jax.jit(jax.vmap(score))(batches), expected, rtol=2e-12)
    gradient = np.asarray(jax.jit(jax.grad(score))(data))
    assert gradient.shape == data.shape
    assert np.all(np.isfinite(gradient))
    epsilon = 1e-5
    plus, minus = data.copy(), data.copy()
    plus[2, 1] += epsilon
    minus[2, 1] -= epsilon
    finite_difference = (float(compiled(plus)) - float(compiled(minus))) / (2 * epsilon)
    np.testing.assert_allclose(gradient[2, 1], finite_difference, rtol=1e-5, atol=1e-8)
    state = functional.batch_state(data, dense, n_clusters=len(unique))
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(state))
    np.testing.assert_allclose(
        functional.evaluate(state, index=index_type.__name__), value,
    )


@pytest.mark.parametrize("index", ["CH", "WB", "XB"])
def test_functional_undefined_scores_and_invalid_dense_partitions(jax_runtime, index):
    _, functional = jax_runtime
    data = np.ones((4, 2))
    labels = np.array([0, 0, 1, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = getattr(cvi, index)().get_cvi(data, labels)
    np.testing.assert_allclose(
        functional.batch_cvi(data, labels, n_clusters=2, index=index),
        expected, equal_nan=True,
    )
    for invalid in (np.array([0, 0, 0, 0]), np.array([0, 0, 1, 3]),
                    np.array([-1, 0, 1, 1])):
        assert np.isnan(functional.batch_cvi(data, invalid, n_clusters=2, index=index))


def test_functional_shape_dtype_and_static_argument_errors(jax_runtime):
    _, functional = jax_runtime
    data = np.ones((4, 2))
    labels = np.array([0, 0, 1, 1])
    for bad_data, bad_labels, k in (
        (data[0], labels, 2), (data, labels[:2], 2),
        (data, labels.astype(float), 2), (data.astype(complex), labels, 2),
        (data, labels, 1), (data, labels, 5),
    ):
        with pytest.raises(ValueError):
            functional.batch_state(bad_data, bad_labels, n_clusters=k)
    with pytest.raises(ValueError, match="JAX batch indices"):
        functional.batch_cvi(data, labels, n_clusters=2, index="DB")


@pytest.mark.parametrize("index_type", INDICES)
def test_jax_object_accepts_device_arrays_and_preserves_input_precision(
    jax_runtime, index_type,
):
    jax, _ = jax_runtime
    data, labels = batch_case("random", np.float64)
    for array in (np.asfortranarray(data), data.astype(np.float16),
                  data.astype(">f8"), jax.device_put(data)):
        expected = index_type().get_cvi(np.asarray(array), labels)
        actual = index_type(backend="jax").get_cvi(array, jax.device_put(labels))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("index", ["CH", "WB", "XB"])
def test_functional_large_offset_and_singleton_partitions(jax_runtime, index):
    _, functional = jax_runtime
    data = np.random.default_rng(90).integers(-20, 20, size=(32, 3)) / 4
    labels = np.arange(32, dtype=np.int32) % 4
    # Powers-of-two denominators make means exactly representable both before
    # and after translation, exposing cancellation-prone implementations.
    np.testing.assert_allclose(
        functional.batch_cvi(data + 2**30, labels, n_clusters=4, index=index),
        getattr(cvi, index)().get_cvi(data, labels), rtol=2e-12, atol=2e-12,
    )
    singletons = np.arange(4)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = getattr(cvi, index)().get_cvi(data[:4], singletons)
    np.testing.assert_allclose(
        functional.batch_cvi(data[:4], singletons, n_clusters=4, index=index),
        expected, equal_nan=True,
    )
