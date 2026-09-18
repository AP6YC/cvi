"""Optional CPU kernels with strict floating-point arithmetic.

Compile numeric loops only. NumPy retains dtype-sensitive means and raw-moment
reductions, so selecting this backend does not silently change float32 means or
integer overflow behavior. Unsupported array types use the reference operation.
"""

import numpy as np
from numba import njit

from .numpy import NumpyBackend


def _supported(array):
    """Accept native real numeric ndarrays supported by Numba."""
    return (
        type(array) is np.ndarray
        and array.dtype.isnative
        and (array.dtype.kind in "biu" or array.dtype in (np.float32, np.float64))
    )


@njit(cache=True, fastmath=False)
def _grouped_rows(labels, n_clusters):
    counts = np.zeros(n_clusters, dtype=np.intp)
    for label in labels:
        counts[label] += 1
    offsets = np.empty(n_clusters + 1, dtype=np.intp)
    offsets[0] = 0
    for cluster in range(n_clusters):
        offsets[cluster + 1] = offsets[cluster] + counts[cluster]
    positions = offsets[:-1].copy()
    order = np.empty(len(labels), dtype=np.intp)
    for row in range(len(labels)):
        label = labels[row]
        order[positions[label]] = row
        positions[label] += 1
    return order, offsets


@njit(cache=True, fastmath=False)
def _compactness(data, order, offsets, centroids):
    values = np.zeros(len(centroids))
    for cluster in range(len(centroids)):
        total = 0.0
        for position in range(offsets[cluster], offsets[cluster + 1]):
            row = order[position]
            for feature in range(data.shape[1]):
                difference = float(data[row, feature]) - centroids[cluster, feature]
                total += difference * difference
        values[cluster] = total
    return values


@njit(cache=True, fastmath=False)
def _centroid_distances(centroids, centroid, squared):
    distances = np.empty(len(centroids))
    for cluster in range(len(centroids)):
        total = 0.0
        for feature in range(centroids.shape[1]):
            difference = centroids[cluster, feature] - centroid[feature]
            total += difference * difference
        distances[cluster] = total if squared else np.sqrt(total)
    return distances


@njit(cache=True, fastmath=False)
def _pairwise_centroid_distances(centroids, squared):
    distances = np.zeros((len(centroids), len(centroids)))
    for ix in range(len(centroids) - 1):
        for jx in range(ix + 1, len(centroids)):
            total = 0.0
            for feature in range(centroids.shape[1]):
                difference = centroids[ix, feature] - centroids[jx, feature]
                total += difference * difference
            value = total if squared else np.sqrt(total)
            distances[ix, jx] = value
            distances[jx, ix] = value
    return distances


@njit(cache=True, fastmath=False, error_model="numpy")
def _silhouette_distances(centroids, compactness, residuals, counts):
    distances = _pairwise_centroid_distances(centroids, True)
    for ix in range(len(centroids)):
        for jx in range(len(centroids)):
            correction = 0.0
            for feature in range(centroids.shape[1]):
                correction += (
                    (centroids[ix, feature] - centroids[jx, feature])
                    * residuals[ix, feature]
                )
            distances[ix, jx] += (compactness[ix] + 2 * correction) / counts[ix]
    return distances


class NumbaBackend(NumpyBackend):
    """Numba loops with NumPy reductions where dtype behavior is significant."""

    name = "numba"
    grouped_rows = staticmethod(_grouped_rows)

    @staticmethod
    def batch_statistics(data, order, offsets, compactness=True):
        if not _supported(data):
            return NumpyBackend.batch_statistics(data, order, offsets, compactness)
        # Keep NumPy's mean reduction order, including its pairwise summation
        # for single-feature groups and its float32 accumulator behavior.
        counts, centroids, values = NumpyBackend.batch_statistics(
            data, order, offsets, compactness=False,
        )
        if compactness:
            values = _compactness(data, order, offsets, centroids)
        return counts, centroids, values

    @staticmethod
    def centroid_distances(centroids, centroid, squared=True):
        if not (_supported(centroids) and _supported(centroid)):
            return NumpyBackend.centroid_distances(centroids, centroid, squared)
        return _centroid_distances(centroids, centroid, squared)

    @staticmethod
    def pairwise_centroid_distances(centroids, squared=True):
        if not _supported(centroids):
            return NumpyBackend.pairwise_centroid_distances(centroids, squared)
        return _pairwise_centroid_distances(centroids, squared)

    @staticmethod
    def silhouette_batch_statistics(data, order, offsets, centroids, compactness):
        if not _supported(data):
            return NumpyBackend.silhouette_batch_statistics(
                data, order, offsets, centroids, compactness,
            )
        raw_compactness = []
        raw_sums = np.zeros_like(centroids)
        residuals = np.zeros_like(centroids)
        for ix in range(len(centroids)):
            subset = data[order[offsets[ix]:offsets[ix + 1]], :]
            raw_compactness.append(np.sum(subset ** 2))
            raw_sums[ix] = np.sum(subset, axis=0)
            residuals[ix] = np.sum(subset - centroids[ix], axis=0)
        distances = _silhouette_distances(
            centroids, np.asarray(compactness), residuals, np.diff(offsets),
        )
        return raw_compactness, raw_sums, distances
