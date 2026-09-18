"""Compatibility adapter for the functional JAX batch implementation."""

import jax
import numpy as np

from .. import jax as functional
from ..modules import _kernels


class JaxBackend:
    """Batch-only adapter; the public object retains host-side summary state."""

    name = "jax"

    def __init__(self):
        functional._require_x64()

    def initialize(self, data, labels, index):
        """Validate and compute a complete state before mutating a CVI object."""
        functional._require_x64()
        functional._check_index(index)
        data, labels = np.asarray(data), np.asarray(labels)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("JAX batch data must be a nonempty two-dimensional array")
        if labels.ndim != 1 or len(labels) != len(data):
            raise ValueError("JAX batch labels must contain one label per sample")
        if labels.dtype.kind not in "iu":
            raise ValueError("JAX batch labels must be integers")
        if data.dtype.kind not in "iuf" or data.dtype.itemsize > 8:
            raise ValueError("JAX batch data must contain real numbers up to 64 bits")
        mapping = dict.fromkeys(int(label) for label in labels)
        mapping = {label: ix for ix, label in enumerate(mapping)}
        if len(mapping) < 2:
            raise ValueError("Batch CVI mode requires at least two unique labels")
        dense = np.fromiter((mapping[int(label)] for label in labels),
                            dtype=np.intp, count=len(labels))
        order, offsets = _kernels.grouped_rows(dense, len(mapping))
        # Preserve the legacy input-dtype mean reductions, particularly for
        # float32/large offsets. The pure functional API computes means on device.
        counts, centers, _ = _kernels.batch_statistics(
            data, order, offsets, compactness=False,
        )
        mean = np.mean(data, axis=0)
        device_state, derived = functional._host_batch(
            np.asarray(data, dtype=np.float64), dense, centers,
            np.asarray(mean, dtype=np.float64), index=index,
        )
        device_state, derived = jax.device_get((device_state, derived))
        result = {
            key: (np.asarray(value).copy() if np.ndim(value) else float(value))
            for key, value in derived.items()
        }
        result.update(
            _n=counts.tolist(), _v=centers, _CP=list(device_state.compactness),
            _G=np.zeros_like(centers), _mu=mean,
            _n_samples=len(data), _dim=data.shape[1], _n_clusters=len(mapping),
            _is_setup=True,
        )
        return mapping, result
