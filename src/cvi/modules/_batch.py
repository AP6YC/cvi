"""Transactional chunk updates for indices with mergeable cluster moments."""

from copy import copy

import numpy as np

from ._kernels import centered_mean


def _validate(index, data, labels):
    data, labels = np.asarray(data), np.asarray(labels)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("Batch update data must have shape (N, n_features)")
    if index._is_setup and data.shape[1] != index._dim:
        raise ValueError(f"Expected {index._dim} features")
    if data.dtype.kind not in "iuf" or data.dtype.itemsize > 8:
        raise ValueError("Batch update data must contain real numbers up to 64 bits")
    if not np.all(np.isfinite(data)):
        raise ValueError("Batch update data must be finite")
    if labels.ndim != 1 or len(labels) != len(data):
        raise ValueError("Batch update labels must contain one integer per sample")
    # An empty Python label list becomes float64; it is still a valid no-op.
    if len(labels) and labels.dtype.kind not in "iu":
        raise ValueError("Batch update labels must contain one integer per sample")
    return np.asarray(data, dtype=np.float64), labels


def _stage(index, data, labels):
    """Build independent sufficient statistics without copying old pair caches."""
    unique, order, offsets = index._prepare_integer_batch_groups(labels)
    mapping = index._label_map.map.copy()
    for label in unique:
        if label not in mapping:
            mapping[label] = len(mapping)
    slots = np.asarray([mapping[label] for label in unique], dtype=np.intp)
    k, d = len(mapping), data.shape[1]
    old_k = index._n_clusters
    counts = np.zeros(k, dtype=np.int64)
    centroids = np.zeros((k, d))
    if old_k:
        counts[:old_k], centroids[:old_k] = index._n, index._v
    # Every supported rebuild replaces derived arrays. The candidate shares
    # old caches only until that rebuild; no shared mutable field is changed.
    candidate = copy(index)
    candidate._label_map = copy(index._label_map)
    candidate._label_map.map = mapping
    candidate._n = counts
    candidate._v = centroids
    candidate._n_clusters = k
    candidate._dim = d
    candidate._is_setup = True
    candidate._merge_batch_statistics(data, order, offsets, slots)
    candidate._n = candidate._n.tolist()
    candidate._n_samples = index._n_samples + len(data)
    if not np.all(np.isfinite(candidate._v)):
        raise ValueError("Batch update statistics exceed float64 range")
    return candidate


def merge_moments(index, data, order, offsets, slots, old_compactness, old_residuals):
    """Merge centered scalar moments into a staged index's counts/centroids."""
    compactness = residuals = None
    if old_compactness is not None:
        compactness = np.zeros(index._n_clusters)
        residuals = np.zeros_like(index._v)
        old_k = len(old_compactness)
        if old_k:
            compactness[:old_k], residuals[:old_k] = old_compactness, old_residuals
    chunk = index._backend.chunk_statistics(
        data, order, offsets, compactness=compactness is not None,
    )
    merged = index._backend.merge_statistics(
        index._n[slots], index._v[slots],
        compactness[slots] if compactness is not None else None,
        residuals[slots] if residuals is not None else None,
        *chunk,
    )
    index._n[slots], index._v[slots] = merged[:2]
    if compactness is not None:
        compactness[slots], residuals[slots] = merged[2:]
        if not all(np.all(np.isfinite(value)) for value in (compactness, residuals)):
            raise ValueError("Batch update statistics exceed float64 range")
    return compactness, residuals


def merge_default(index, data, order, offsets, slots):
    """Populate shared compactness or centroid-only state on a staged index."""
    compactness, residuals = merge_moments(
        index, data, order, offsets, slots,
        index._CP if index._uses_compactness_stats else None,
        index._G if index._uses_compactness_stats else None,
    )
    if index._uses_compactness_stats:
        index._CP, index._G = compactness.tolist(), residuals
        mean = centered_mean(data)
        index._mu = (
            mean if index._n_samples == 0 else
            index._mu + (mean - index._mu) * (len(data) / (index._n_samples + len(data)))
        )
        if not np.all(np.isfinite(index._mu)):
            raise ValueError("Batch update statistics exceed float64 range")
    else:
        # PS only uses counts and centroids; keep its common empty fields.
        index._CP = []
        index._G = np.zeros((0, index._dim))


def update_batch(index, data, labels):
    """Validate, aggregate, rebuild once, and commit the complete update."""
    data, labels = _validate(index, data, labels)
    if len(data) == 0:
        return float(index.criterion_value)
    try:
        with np.errstate(over="raise", invalid="raise"):
            candidate = _stage(index, data, labels)
    except FloatingPointError as error:
        raise ValueError("Batch update statistics exceed float64 range") from error
    candidate._rebuild_after_operation()
    candidate._evaluate()
    index.__dict__.update(candidate.__dict__)
    return float(index.criterion_value)
