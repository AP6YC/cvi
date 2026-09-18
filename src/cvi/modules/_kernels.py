"""Array-only NumPy kernels shared by the CVI implementations.

Labels passed here are dense internal integers. Public validation, external
label mapping, and object mutation belong to the caller. Grouping is stable
so that each cluster's reductions retain the original sample order.
"""

import numpy as np


def grouped_rows(labels: np.ndarray, n_clusters: int):
    """Return sample indices grouped by label and cluster slice boundaries."""
    order = np.argsort(labels, kind="stable")
    offsets = np.empty(n_clusters + 1, dtype=np.intp)
    offsets[0] = 0
    offsets[1:] = np.cumsum(np.bincount(labels, minlength=n_clusters))
    return order, offsets


def batch_statistics(data, order, offsets, compactness=True):
    """Compute counts, centroids, and optionally centered sums of squares.

    All clusters must be nonempty. Means use the input dtype's NumPy reduction
    before assignment to float64, matching the existing batch implementations.
    Compactness uses centered differences rather than subtracting raw moments,
    which avoids cancellation for clusters far from the origin.
    """
    counts = np.diff(offsets)
    centroids = np.zeros((len(counts), data.shape[1]))
    squared_errors = np.zeros(len(counts))
    for ix in range(len(counts)):
        subset = data[order[offsets[ix]:offsets[ix + 1]], :]
        centroids[ix] = np.mean(subset, axis=0)
        if compactness:
            squared_errors[ix] = np.sum((subset - centroids[ix]) ** 2)
    return counts, centroids, squared_errors


def centroid_distances(centroids, centroid, squared=True):
    """Distances to one centroid, without a cancellation-prone Gram matrix."""
    distances = np.sum((centroids - centroid) ** 2, axis=1)
    return distances if squared else np.sqrt(distances)


def pairwise_centroid_distances(centroids, squared=True):
    """Symmetric distances with zero diagonal and at most O(K d) scratch."""
    n_clusters = len(centroids)
    distances = np.zeros((n_clusters, n_clusters))
    for ix in range(n_clusters - 1):
        values = centroid_distances(
            centroids[ix + 1:], centroids[ix], squared=squared,
        )
        distances[ix, ix + 1:] = values
        distances[ix + 1:, ix] = values
    return distances


def silhouette_batch_statistics(data, order, offsets, centroids, compactness):
    """Return cSIL's raw moments and cluster-to-centroid mean distances.

    S[i, j] is the mean squared distance from cluster i's samples to centroid
    j. Its centered expansion avoids both an N-by-K distance matrix and raw
    moment cancellation. The residual term accounts for rounding in the mean,
    including float32 inputs; it must not be assumed to be exactly zero.
    Raw moments retain their original reduction dtype for subsequent updates.
    """
    n_clusters = len(centroids)
    raw_compactness = []
    raw_sums = np.zeros_like(centroids)
    dissimilarities = pairwise_centroid_distances(centroids)
    for ix in range(n_clusters):
        subset = data[order[offsets[ix]:offsets[ix + 1]], :]
        count = len(subset)
        raw_compactness.append(np.sum(subset ** 2))
        raw_sums[ix] = np.sum(subset, axis=0)
        residual = np.sum(subset - centroids[ix], axis=0)
        correction = 2 * np.sum(
            (centroids[ix] - centroids) * residual, axis=1,
        )
        dissimilarities[ix] += (compactness[ix] + correction) / count
    return raw_compactness, raw_sums, dissimilarities
