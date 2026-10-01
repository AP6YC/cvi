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
    use_compactness = index._uses_compactness_stats
    compactness = residuals = None
    if use_compactness:
        compactness = np.zeros(k)
        residuals = np.zeros((k, d))
        if old_k:
            compactness[:old_k], residuals[:old_k] = index._CP, index._G
    chunk = index._backend.chunk_statistics(
        data, order, offsets, compactness=use_compactness,
    )
    merged = index._backend.merge_statistics(
        counts[slots], centroids[slots],
        compactness[slots] if use_compactness else None,
        residuals[slots] if use_compactness else None,
        *chunk,
    )
    counts[slots], centroids[slots] = merged[:2]
    if use_compactness:
        compactness[slots], residuals[slots] = merged[2:]

    # Every supported rebuild replaces derived arrays. The candidate shares
    # old caches only until that rebuild; no shared mutable field is changed.
    candidate = copy(index)
    candidate._label_map = copy(index._label_map)
    candidate._label_map.map = mapping
    candidate._n = counts.tolist()
    candidate._v = centroids
    candidate._n_clusters = k
    candidate._n_samples = index._n_samples + len(data)
    candidate._dim = d
    candidate._is_setup = True
    if use_compactness:
        candidate._CP = compactness.tolist()
        candidate._G = residuals
        mean = centered_mean(data)
        candidate._mu = (
            mean if index._n_samples == 0 else
            index._mu + (mean - index._mu) * (len(data) / candidate._n_samples)
        )
        if not all(np.all(np.isfinite(value)) for value in
                   (compactness, residuals, candidate._mu)):
            raise ValueError("Batch update statistics exceed float64 range")
    else:
        # PS only uses counts and centroids; keep its common empty fields.
        candidate._CP = []
        candidate._G = np.zeros((0, d))
    if not np.all(np.isfinite(centroids)):
        raise ValueError("Batch update statistics exceed float64 range")
    return candidate


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
