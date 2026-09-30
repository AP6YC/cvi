"""Default NumPy implementation of the shared numerical operations."""

from ..modules import _kernels


class NumpyBackend:
    """Stateless, pickleable adapter for the reference kernels."""

    name = "numpy"
    grouped_rows = staticmethod(_kernels.grouped_rows)
    batch_statistics = staticmethod(_kernels.batch_statistics)
    centroid_distances = staticmethod(_kernels.centroid_distances)
    pairwise_centroid_distances = staticmethod(_kernels.pairwise_centroid_distances)
    minimum_off_diagonal = staticmethod(_kernels.minimum_off_diagonal)
    silhouette_batch_statistics = staticmethod(_kernels.silhouette_batch_statistics)
