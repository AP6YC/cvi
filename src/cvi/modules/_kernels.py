"""Array-only NumPy kernels shared by the CVI implementations.

Labels passed here are dense internal integers. Public validation, external
label mapping, and object mutation belong to the caller. Grouping is stable
so that each cluster's reductions retain the original sample order.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


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


def centered_mean(data):
    """Reduce float64 data about one observation to limit offset roundoff."""
    origin = data[0]
    return origin + np.mean(data - origin, axis=0)


def chunk_statistics(data, order, offsets, compactness=True):
    """Summarize nonempty float64 groups, retaining centered residual sums.

    Unlike legacy batch initialization, chunk updates always accumulate in
    float64. Residuals account for rounded centroids when summaries are merged
    or subsequently updated one sample at a time.
    """
    counts = np.diff(offsets)
    centroids = np.empty((len(counts), data.shape[1]))
    squared_errors = np.zeros(len(counts)) if compactness else None
    residuals = np.zeros_like(centroids) if compactness else None
    for ix in range(len(counts)):
        subset = data[order[offsets[ix]:offsets[ix + 1]]]
        centroids[ix] = centered_mean(subset)
        if compactness:
            centered = subset - centroids[ix]
            squared_errors[ix] = np.sum(centered ** 2)
            residuals[ix] = np.sum(centered, axis=0)
    return counts, centroids, squared_errors, residuals


def merge_statistics(counts, centroids, compactness, residuals,
                     chunk_counts, chunk_centroids, chunk_compactness,
                     chunk_residuals):
    """Combine aligned summaries without modifying either input.

    Zero old counts indicate new clusters. Translate both centered moments to
    the rounded combined centroid, including their residual corrections.
    """
    combined_counts = counts + chunk_counts
    combined_centroids = chunk_centroids.copy()
    existing = counts > 0
    combined_centroids[existing] = (
        centroids[existing]
        + (chunk_centroids[existing] - centroids[existing])
        * (chunk_counts[existing] / combined_counts[existing])[:, None]
    )
    combined_compactness = combined_residuals = None
    if compactness is not None:
        combined_compactness = chunk_compactness.copy()
        combined_residuals = chunk_residuals.copy()
        left_shift = centroids[existing] - combined_centroids[existing]
        right_shift = chunk_centroids[existing] - combined_centroids[existing]
        combined_compactness[existing] += (
            compactness[existing]
            + counts[existing] * np.sum(left_shift ** 2, axis=1)
            + chunk_counts[existing] * np.sum(right_shift ** 2, axis=1)
            + 2 * np.sum(left_shift * residuals[existing], axis=1)
            + 2 * np.sum(right_shift * chunk_residuals[existing], axis=1)
        )
        combined_residuals[existing] += (
            residuals[existing]
            + counts[existing, None] * left_shift
            + chunk_counts[existing, None] * right_shift
        )
    return (combined_counts, combined_centroids, combined_compactness,
            combined_residuals)


def centroid_distances(centroids, centroid, squared=True):
    """Distances to one centroid, without a cancellation-prone Gram matrix."""
    distances = np.sum((centroids - centroid) ** 2, axis=1)
    return distances if squared else np.sqrt(distances)


def pairwise_centroid_distances(centroids, squared=True):
    """Symmetric distances using SciPy's compiled direct-distance kernels."""
    n_clusters = len(centroids)
    if n_clusters < 2:
        return np.zeros((n_clusters, n_clusters))
    metric = "sqeuclidean" if squared else "euclidean"
    return squareform(pdist(centroids, metric=metric))


def minimum_off_diagonal(values):
    """Return the minimum value outside a square matrix's diagonal.

    The diagonal is replaced only for the duration of the reduction.  This
    avoids allocating triangle indices or a full boolean mask on every CVI
    evaluation while leaving the caller's matrix unchanged.
    """
    diagonal = np.diag(values).copy()
    try:
        np.fill_diagonal(values, np.inf)
        return np.min(values)
    finally:
        np.fill_diagonal(values, diagonal)


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
    residuals = np.zeros_like(centroids)
    dissimilarities = pairwise_centroid_distances(centroids)
    for ix in range(n_clusters):
        subset = data[order[offsets[ix]:offsets[ix + 1]], :]
        count = len(subset)
        raw_compactness.append(np.sum(subset ** 2))
        raw_sums[ix] = np.sum(subset, axis=0)
        residual = np.sum(subset - centroids[ix], axis=0)
        residuals[ix] = residual
        correction = 2 * np.sum(
            (centroids[ix] - centroids) * residual, axis=1,
        )
        dissimilarities[ix] += (compactness[ix] + correction) / count
    return raw_compactness, raw_sums, dissimilarities, residuals


def silhouette_from_moments(centroids, compactness, residuals, counts):
    """Mean squared cluster-to-centroid distances from centered summaries."""
    values = pairwise_centroid_distances(centroids)
    for i in range(len(centroids)):
        correction = 2 * np.sum((centroids[i] - centroids) * residuals[i], axis=1)
        values[i] += (compactness[i] + correction) / counts[i]
    return values


def covariance_statistics(data, order, offsets):
    """Float64 means and unregularized sample covariances of nonempty groups."""
    counts = np.diff(offsets)
    centroids = np.empty((len(counts), data.shape[1]))
    covariances = np.zeros((len(counts), data.shape[1], data.shape[1]))
    for i, count in enumerate(counts):
        subset = np.asarray(data[order[offsets[i]:offsets[i + 1]]], dtype=np.float64)
        shifted = subset - subset[0]
        mean = np.mean(shifted, axis=0)
        centroids[i] = subset[0] + mean
        if count > 1:
            centered = shifted - mean
            covariances[i] = (centered.T @ centered) / (count - 1)
    return counts, centroids, covariances


def merge_covariances(counts, centroids, covariances,
                      chunk_counts, chunk_centroids, chunk_covariances):
    """Merge unregularized covariances by combining their centered scatter."""
    combined_counts = counts + chunk_counts
    combined_centroids = chunk_centroids.copy()
    combined_covariances = chunk_covariances.copy()
    existing = counts > 0
    a, b, n = counts[existing], chunk_counts[existing], combined_counts[existing]
    difference = chunk_centroids[existing] - centroids[existing]
    combined_centroids[existing] = centroids[existing] + difference * (b / n)[:, None]
    combined_covariances[existing] = (
        (a - 1)[:, None, None] * covariances[existing]
        + (b - 1)[:, None, None] * chunk_covariances[existing]
        + (a * (b / n))[:, None, None]
        * difference[:, :, None] * difference[:, None, :]
    ) / (n - 1)[:, None, None]
    return combined_counts, combined_centroids, combined_covariances
