"""Compatibility adapter for the functional JAX implementation."""

import jax
import numpy as np

from .. import jax as functional
from ..modules import _kernels


class JaxBackend:
    """Host label validation with optional device-resident streaming statistics."""

    name = "jax"

    def __init__(self):
        functional._require_x64()

    def initialize(self, data, labels, index, *, capacity=None):
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
        if capacity is not None:
            if len(mapping) > capacity:
                raise ValueError("Cluster count exceeds capacity")
            if not np.all(np.isfinite(data)):
                raise ValueError("Streaming data must be finite")
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
        if capacity is not None:
            stream = functional.stream_from_batch(
                device_state, capacity=capacity, index=index,
            )
            return mapping, self._stream_updates(stream, index, len(mapping), len(data))
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

    @staticmethod
    def _stream_updates(state, index, n_clusters, n_samples):
        derived = functional._stream_derived(state, index)
        # Synchronize the scalar before committing; summary arrays stay on device.
        value = float(derived["criterion_value"])
        return dict(
            derived, criterion_value=value, _stream_state=state,
            _n=state.counts, _v=state.centroids, _CP=state.compactness,
            _G=state.residuals, _mu=state.mean, _n_samples=n_samples,
            _dim=state.centroids.shape[1], _n_clusters=n_clusters, _is_setup=True,
        )

    def advance(self, obj, data, labels, *, return_history, single=False):
        """Preflight the whole chunk and return a commit without mutating obj."""
        functional._require_x64()
        if not isinstance(return_history, bool):
            raise ValueError("return_history must be a boolean")
        data, labels = np.asarray(data), np.asarray(labels)
        if data.ndim != 2 or data.shape[1] == 0:
            raise ValueError("Streaming data must have shape (N, n_features)")
        if obj._is_setup and data.shape[1] != obj._dim:
            raise ValueError(f"Expected {obj._dim} features")
        if data.dtype.kind not in "iuf" or data.dtype.itemsize > 8:
            raise ValueError("Streaming data must contain real numbers up to 64 bits")
        if not np.all(np.isfinite(data)):
            raise ValueError("Streaming data must be finite")
        if labels.ndim != 1 or len(labels) != len(data) or labels.dtype.kind not in "iu":
            raise ValueError("Streaming labels must contain one integer per sample")
        mapping = obj._label_map.map.copy()
        slots = np.empty(len(labels), dtype=np.int64)
        for i, label in enumerate(labels):
            label = int(label)
            if label not in mapping:
                if len(mapping) == obj.capacity:
                    raise ValueError("Cluster count exceeds capacity")
                mapping[label] = len(mapping)
            slots[i] = mapping[label]
        if len(data) == 0:
            output = np.empty(0, dtype=np.float64) if return_history else obj.criterion_value
            return mapping, {}, output
        index = obj.info.name_short
        state = obj.stream_state
        if state is None:
            state = functional.empty_stream(
                capacity=obj.capacity, n_features=data.shape[1], index=index,
            )
        if single:
            state, result = functional.stream_update(state, data[0], slots[0], index=index)
        else:
            state, result = functional.stream_chunk(
                state, data, slots, index=index, return_history=return_history,
            )
        output = np.asarray(result).copy() if return_history else float(result)
        updates = self._stream_updates(state, index, len(mapping), obj._n_samples + len(data))
        return mapping, updates, output
