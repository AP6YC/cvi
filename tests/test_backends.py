"""Optional dependency, serialization, and backend selection contracts."""

import copy
import importlib
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.backends import get_backend


SUPPORTED = [index for index in cvi.MODULES if index._supports_numba]


@pytest.mark.parametrize("index_type", cvi.MODULES)
def test_default_backend_and_invalid_selection(index_type):
    assert index_type().backend == "numpy"
    with pytest.raises(ValueError, match="backend must"):
        index_type(backend="unknown")


@pytest.mark.parametrize("index_type", [cvi.CONN, cvi.rCIP])
def test_unsupported_backend_is_explicit(index_type):
    with pytest.raises(NotImplementedError, match="does not support the numba"):
        index_type(backend="numba")


def test_numpy_import_and_execution_without_numba():
    # A fresh process proves that importing CVI does not eagerly load Numba,
    # including through CONN's ART dependency. Block Numba even if installed.
    code = '''
import importlib.abc
import sys
class NoNumba(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numba" or fullname.startswith("numba."):
            raise ModuleNotFoundError("Numba blocked for test", name="numba")
sys.meta_path.insert(0, NoNumba())
import numpy as np
import src.cvi as cvi
data = np.array([[0., 0.], [1., 1.], [4., 4.], [5., 5.]])
labels = np.array([10, 10, 20, 20])
for index_type in cvi.MODULES:
    if index_type is not cvi.CONN:
        assert np.isfinite(index_type().get_cvi(data, labels))
assert "numba" not in sys.modules
assert "src.cvi.backends.numba" not in sys.modules
try:
    cvi.XB(backend="numba")
except ImportError as error:
    assert 'pip install "cvi[numba]"' in str(error)
else:
    raise AssertionError("Missing dependency must not silently select NumPy")
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("index_type", SUPPORTED)
def test_backend_is_per_instance_and_survives_reset_and_serialization(
    index_type, backend,
):
    chosen = index_type(backend=backend)
    assert index_type().backend == "numpy"
    with pytest.raises(AttributeError):
        chosen.backend = "numpy"
    chosen.get_cvi(np.array([0.25, 0.75]), 42)
    chosen.remove(np.array([0.25, 0.75]), 42)
    assert chosen.backend == backend
    assert not chosen._is_setup
    for restored in (copy.deepcopy(chosen), pickle.loads(pickle.dumps(chosen))):
        assert restored.backend == backend
        data = np.array([[0., 0.], [1., 1.], [4., 4.], [5., 5.]])
        labels = np.array([10, 10, 20, 20])
        np.testing.assert_allclose(
            restored.get_cvi(data, labels), index_type().get_cvi(data, labels),
            rtol=2e-12, atol=2e-12,
        )


@pytest.mark.parametrize("dtype", [np.float16, np.dtype(">f8")])
@pytest.mark.parametrize("index_type", SUPPORTED)
def test_unsupported_numba_dtypes_preserve_numpy_behavior(index_type, dtype):
    pytest.importorskip("numba")
    data = np.array([[0., 0.], [1., 1.], [4., 4.], [5., 5.]], dtype=dtype)
    labels = np.array([10, 10, 20, 20])
    accelerated, reference = index_type(backend="numba"), index_type()
    np.testing.assert_allclose(
        accelerated.get_cvi(data, labels), reference.get_cvi(data, labels),
        rtol=2e-12, atol=2e-12,
    )
    np.testing.assert_allclose(
        accelerated.get_cvi(data[0], 99), reference.get_cvi(data[0], 99),
        rtol=2e-12, atol=2e-12,
    )


def test_numba_executes_compiled_kernels():
    pytest.importorskip("numba")
    from src.cvi.backends import numba as implementation

    backend = get_backend("numba")
    data = np.arange(60., dtype=float).reshape(20, 3)
    labels = np.arange(20, dtype=np.intp) % 3
    order, offsets = backend.grouped_rows(labels, 3)
    _, centers, compactness = backend.batch_statistics(data, order, offsets)
    backend.centroid_distances(centers, centers[0])
    backend.pairwise_centroid_distances(centers)
    backend.silhouette_batch_statistics(data, order, offsets, centers, compactness)
    for name in ("_grouped_rows", "_compactness", "_centroid_distances",
                 "_pairwise_centroid_distances", "_silhouette_distances"):
        dispatcher = getattr(implementation, name)
        assert dispatcher.nopython_signatures, name
        assert dispatcher.targetoptions["fastmath"] is False


def test_numba_grouping_is_stable_for_many_interleaved_labels():
    pytest.importorskip("numba")
    labels = np.random.default_rng(9).integers(100, size=10000, dtype=np.intp)
    expected = get_backend("numpy").grouped_rows(labels, 100)
    actual = get_backend("numba").grouped_rows(labels, 100)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


def test_pre_backend_pickles_default_to_numpy():
    index = cvi.XB()
    data = np.array([[0., 0.], [1., 1.], [4., 4.], [5., 5.]])
    labels = np.array([10, 10, 20, 20])
    index.get_cvi(data, labels)
    del index._backend  # State layout used before optional backends existed.
    restored = pickle.loads(pickle.dumps(index))
    reference = cvi.XB()
    reference.get_cvi(data, labels)
    assert restored.backend == "numpy"
    assert restored.get_cvi(data[0], 10) == reference.get_cvi(data[0], 10)


def test_legacy_conn_adapter_paths_remain_resolvable():
    module = importlib.import_module("src.cvi.modules.CONN")
    adapters = importlib.import_module("src.cvi.modules._conn_art")
    for name in ("_CONNFuzzyART", "_CONNSimpleARTMAP"):
        assert getattr(module, name) is getattr(adapters, name)


@pytest.mark.parametrize("features", [1, 8, 33])
def test_large_offset_batch_with_long_reductions(features):
    pytest.importorskip("numba")
    rng = np.random.default_rng(45)
    data = rng.normal(size=(4000, features)) + 1e12
    labels = np.arange(len(data), dtype=np.intp) % 7
    compiled = get_backend("numba")
    reference = get_backend("numpy")
    order, offsets = reference.grouped_rows(labels, 7)
    counts, centers, compactness = compiled.batch_statistics(data, order, offsets)
    expected_counts, expected_centers, expected_cp = reference.batch_statistics(
        data, order, offsets,
    )
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_allclose(compactness, expected_cp, rtol=2e-12)
    actual = compiled.silhouette_batch_statistics(
        data, order, offsets, centers, compactness,
    )
    expected = reference.silhouette_batch_statistics(
        data, order, offsets, expected_centers, expected_cp,
    )
    for result, expected_result in zip(actual, expected):
        np.testing.assert_allclose(result, expected_result, rtol=2e-12, atol=2e-12)
