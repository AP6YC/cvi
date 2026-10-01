"""Independent numerical oracles for cSIL and rCIP chunk updates."""

import pickle

import numpy as np
import pytest

import src.cvi as cvi


def assert_csil_distances(index, data, labels):
    for label, i in index._label_map.map.items():
        group = data[labels == label]
        centered = group - index._v[i]
        np.testing.assert_allclose(index._centered_CP[i], np.sum(centered ** 2),
                                   rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(index._residuals[i], np.sum(centered, axis=0),
                                   rtol=2e-12, atol=2e-12)
        for j, center in enumerate(index._v):
            expected = np.mean(np.sum((group - center) ** 2, axis=1))
            np.testing.assert_allclose(index._S[i, j], expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("initial", ["batch", "stream", "chunk"])
def test_csil_large_offset_mixed_updates(initial, backend):
    rng = np.random.default_rng(147)
    data = 1e12 + rng.integers(-16, 17, size=(48, 3)) / 8
    labels = np.arange(len(data)) % 3
    index = cvi.cSIL(backend=backend)
    if initial == "batch":
        index.get_cvi(data[:12], labels[:12])
    elif initial == "stream":
        for x, y in zip(data[:12], labels[:12]):
            index.get_cvi(x, int(y))
    else:
        index.update_batch(data[:12], labels[:12])
    for stop in (19, 33, 48):
        start = index._n_samples
        index.update_batch(data[start:stop], labels[start:stop])
        assert_csil_distances(index, data[:stop], labels[:stop])
    index.remove(data[0], int(labels[0]))
    data, labels = data[1:], labels[1:]
    assert_csil_distances(index, data, labels)
    index.merge(0, 1)
    labels[labels == 1] = 0
    assert_csil_distances(index, data, labels)
    subset = np.flatnonzero(labels == 0)[:2]
    # Two grid points have an exactly representable mean, so the public split
    # summary loses no residual information in this numerical control.
    centroid = data[subset].mean(axis=0)
    compactness = np.sum((data[subset] - centroid) ** 2)
    index.split(0, 999, 2, centroid, compactness=compactness)
    labels[subset] = 999
    assert_csil_distances(index, data, labels)
    index.get_cvi(data[0], int(labels[0]))
    data, labels = np.vstack((data, data[0])), np.r_[labels, labels[0]]
    assert_csil_distances(index, data, labels)
    index.update_batch(data[:3], labels[:3])
    assert_csil_distances(index, np.vstack((data, data[:3])), np.r_[labels, labels[:3]])


@pytest.mark.parametrize("initial", ["empty", "batch", "stream"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rcip_translation_invariant_covariance(initial, dtype):
    rng = np.random.default_rng(146)
    offset = 1e4 if dtype == np.float32 else 1e9
    data = (offset + rng.normal(size=(173, 3))).astype(dtype)
    labels = np.arange(len(data)) % 4
    index = cvi.rCIP()
    start = 0
    if initial != "empty":
        start = 21
        if initial == "batch":
            index.get_cvi(data[:start], labels[:start])
        else:
            for x, y in zip(data[:start], labels[:start]):
                index.get_cvi(x, int(y))
    while start < len(data):
        stop = min(start + 17, len(data))
        index.update_batch(data[start:stop], labels[start:stop])
        for label, i in index._label_map.map.items():
            group = np.asarray(data[:stop][labels[:stop] == label], dtype=float)
            shifted = group - group[0]
            expected = np.cov(shifted, rowvar=False) + index._delta_term
            # Float64 centroids at 1e9 are quantized to ~1e-7. Covariance
            # comparisons allow that centroid precision, not raw-moment loss.
            np.testing.assert_allclose(index._sigma[:, :, i], expected,
                                       rtol=1e-6, atol=1e-6)
        start = stop
    assert np.isfinite(index.criterion_value)


def test_rcip_batch_uses_centered_float64_covariances():
    rng = np.random.default_rng(147)
    data = (1e6 + rng.normal(size=(151, 4))).astype(np.float32)
    labels = np.arange(len(data)) % 3
    index = cvi.rCIP()
    index.get_cvi(data, labels)
    for label, i in index._label_map.map.items():
        shifted = data[labels == label].astype(float) - data[labels == label][0]
        expected = np.cov(shifted, rowvar=False) + index._delta_term
        np.testing.assert_allclose(index._sigma[:, :, i], expected, rtol=2e-12, atol=2e-12)


def test_rcip_regularizes_once_and_preserves_singletons():
    index = cvi.rCIP()
    for _ in range(20):
        index.update_batch([[1., 2., 3.], [1., 2., 3.]], [10, 10])
    index.update_batch([[2., 3., 4.]], [90])
    for i in range(2):
        np.testing.assert_array_equal(index._sigma[:, :, i], index._delta_term)
    index.update_batch([[4., 5., 6.]], [90])
    expected = np.full((3, 3), 2.) + index._delta_term
    np.testing.assert_allclose(index._sigma[:, :, 1], expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("initialized", [False, True])
def test_legacy_csil_checkpoint_can_continue_with_chunks(initialized):
    index = cvi.cSIL()
    data = np.arange(36, dtype=float).reshape(12, 3) / 7
    labels = np.arange(len(data)) % 2
    if initialized:
        index.get_cvi(data[:6], labels[:6])
    del index._centered_CP
    del index._residuals
    restored = pickle.loads(pickle.dumps(index))
    start = 6 if initialized else 0
    restored.update_batch(data[start:], labels[start:])
    assert_csil_distances(restored, data, labels)


@pytest.mark.parametrize("index_type", [cvi.cSIL, cvi.rCIP])
def test_failed_distribution_update_preserves_specialized_state(index_type, monkeypatch):
    index = index_type()
    index.update_batch([[0., 1.], [1., 0.], [2., 3.]], [10, 10, 20])
    snapshot = pickle.dumps(index)

    def fail(candidate):
        if index_type is cvi.cSIL:
            candidate._centered_CP[0] = 999
            candidate._residuals[:] = 999
        else:
            candidate._sigma[:] = 999
        raise ValueError("failure after moment merge")

    monkeypatch.setattr(index_type, "_rebuild_after_operation", fail)
    with pytest.raises(ValueError, match="after moment merge"):
        index.update_batch([[9., 8.]], [999])
    assert pickle.dumps(index) == snapshot
