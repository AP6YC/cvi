"""
Centroid-based Silhouette (cSIL) Cluster Validity Index.

References
----------
1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
2. P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, vol. 20, pp. 53-65, 1987.
3. M. Rawashdeh and A. Ralescu, "Center-wise intra-inter silhouettes," in Scalable Uncertainty Management, E. Hüllermeier, S. Link, T. Fober et al., Eds. Berlin, Heidelberg: Springer, 2012, pp. 406-419.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# cSIL object definition
class cSIL(_base.CVI):
    """
    Centroid-based Silhouette (cSIL) Cluster Validity Index.

    References
    ----------
    1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    2. P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, vol. 20, pp. 53-65, 1987.
    3. M. Rawashdeh and A. Ralescu, "Center-wise intra-inter silhouettes," in Scalable Uncertainty Management, E. Hüllermeier, S. Link, T. Fober et al., Eds. Berlin, Heidelberg: Springer, 2012, pp. 406-419.
    """

    info = _base.CVIInfo(
        name="Centroid-based Silhouette",
        name_short="cSIL",
        index_min=-1.0,
        index_max=1.0,
        optimality="max",
        batch=True,
        incremental=True,
        merge=True,
        remove=True,
        split=True,
        backends=("numpy", "numba"),
    )
    _supports_numba = True
    _supports_remove_merge = True

    def __init__(self, *, backend="numpy"):
        """
        Centroid-based Silhouette (cSIL) initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

        # cSIL-specific initialization
        self._S = np.empty([0, 0])   # n_clusters x dim
        self._sil_coefs = []         # dim

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Centroid-based Silhouette (cSIL) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

    @staticmethod
    def _cluster_to_centroids(raw_compactness, raw_sum, count, centroids):
        """Mean squared distances from one cluster to every centroid."""
        centroid_norms = np.einsum("ij,ij->i", centroids, centroids)
        return (
            raw_compactness
            + count * centroid_norms
            - 2 * (centroids @ raw_sum)
        ) / count

    @staticmethod
    def _clusters_to_centroid(
        raw_compactness, raw_sums, counts, centroid,
    ):
        """Mean squared distances from every cluster to one centroid."""
        raw_compactness = np.asarray(raw_compactness)
        counts = np.asarray(counts)
        centroid_norm = np.inner(centroid, centroid)
        return (
            raw_compactness
            + counts * centroid_norm
            - 2 * (raw_sums @ centroid)
        ) / counts

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Centroid-based Silhouette (cSIL) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self._label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self._n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self._n_samples == 0:
            self._setup(sample)

        # IF NEW CLUSTER LABEL
        # Correct for python 0-indexing
        if i_label > self._n_clusters - 1:
            n_new = 1
            v_new = sample
            CP_new = np.inner(sample, sample)
            G_new = sample

            # Compute S_new
            if self._n_clusters == 0:
                S_new = np.zeros([1, 1])
            else:
                S_new = np.zeros((self._n_clusters + 1, self._n_clusters + 1))
                S_new[0:self._n_clusters, 0:self._n_clusters] = self._S
                S_row_new = np.zeros(self._n_clusters + 1)
                S_col_new = np.zeros(self._n_clusters + 1)
                S_col_new[:-1] = self._cluster_to_centroids(
                    CP_new, G_new, n_new, self._v,
                )
                S_row_new[:-1] = self._clusters_to_centroid(
                    self._CP, self._G, self._n, v_new,
                )
                S_col_new[i_label] = 0
                S_row_new[i_label] = S_col_new[i_label]
                S_new[i_label, :] = S_col_new
                S_new[:, i_label] = S_row_new

            # Update 1-D parameters with list appends
            self._n_clusters += 1
            self._n.append(n_new)
            self._CP.append(CP_new)

            # Update 2-D parameters with numpy vstacks
            self._v = np.vstack([self._v, v_new])
            self._G = np.vstack([self._G, G_new])
            self._S = S_new

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self._n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self._v[i_label, :]
                + (1 / n_new) * sample
            )
            # delta_v = self._v[i_label, :] - v_new
            # diff_x_v = sample - v_new
            CP_new = (
                self._CP[i_label]
                + np.inner(sample, sample)
            )
            G_new = (
                self._G[i_label, :]
                + sample
            )
            # Compute S_new
            S_col_new = self._cluster_to_centroids(
                CP_new, G_new, n_new, self._v,
            )
            S_row_new = self._clusters_to_centroid(
                self._CP, self._G, self._n, v_new,
            )

            diagonal = (
                CP_new
                + n_new * np.inner(v_new, v_new)
                - 2 * np.inner(G_new, v_new)
            ) / n_new
            S_col_new[i_label] = diagonal
            S_row_new[i_label] = diagonal

            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._CP[i_label] = CP_new
            self._G[i_label, :] = G_new

            # self._S[:, i_label] = S_col_new
            # self._S[i_label, :] = S_row_new
            self._S[i_label, :] = S_col_new
            self._S[:, i_label] = S_row_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Centroid-based Silhouette (cSIL) CVI.
        """

        order, offsets = self._setup_batch_statistics(data, labels)
        self._CP, self._G, self._S = self._backend.silhouette_batch_statistics(
            data, order, offsets, self._v, self._CP,
        )

    def _delete_cluster(self, label: int, i_label: int):
        """Delete one cSIL cluster and compact its internal label."""

        self._CP = self._delete_vector_entry(self._CP, i_label)
        self._G = np.delete(self._G, i_label, axis=0)
        super()._delete_cluster(label, i_label)

    def _remove(self, sample: np.ndarray, label: int, i_label: int):
        """Remove a sample from cSIL's zero-centered raw moments."""

        n_old = self._n[i_label]
        v_old = self._v[i_label, :].copy()
        n_samples_new = self._n_samples - 1

        if n_old == 1:
            self._validate_singleton_removal(sample, v_old)

            self._delete_cluster(label, i_label)
            self._n_samples = n_samples_new

            if n_samples_new == 0:
                self._clear_common_state()

            self._rebuild_after_operation()
            return

        n_new = n_old - 1
        G_new = self._G[i_label, :] - sample
        v_new = G_new / n_new
        raw_CP_new = self._CP[i_label] - np.inner(sample, sample)
        centered_CP_new = raw_CP_new - n_new * np.inner(v_new, v_new)
        centered_CP_new = self._nonnegative_or_error(
            centered_CP_new,
            max(abs(self._CP[i_label]), abs(raw_CP_new)),
            "cluster compactness",
        )
        raw_CP_new = centered_CP_new + n_new * np.inner(v_new, v_new)

        self._n[i_label] = n_new
        self._v[i_label, :] = v_new
        self._CP[i_label] = raw_CP_new
        self._G[i_label, :] = G_new
        self._n_samples = n_samples_new
        self._rebuild_after_operation()

    def _merge(
        self,
        target_label: int,
        source_label: int,
        target_i: int,
        source_i: int,
    ):
        """Merge two cSIL raw-moment summaries."""

        n_new = self._n[target_i] + self._n[source_i]
        G_new = self._G[target_i, :] + self._G[source_i, :]
        CP_new = self._CP[target_i] + self._CP[source_i]

        self._n[target_i] = n_new
        self._v[target_i, :] = G_new / n_new
        self._CP[target_i] = CP_new
        self._G[target_i, :] = G_new
        self._delete_cluster(source_label, source_i)
        self._rebuild_after_operation()

    def _split(
        self,
        new_label: int,
        retained_i: int,
        count: int,
        centroid: np.ndarray,
        compactness,
        covariance,
    ):
        """Split centered statistics from cSIL's raw moments."""

        if compactness is None:
            raise ValueError("cSIL split requires compactness")

        n_parent = self._n[retained_i]
        n_remainder = n_parent - count
        G_split = count * centroid
        raw_CP_split = (
            compactness + count * np.inner(centroid, centroid)
        )
        G_remainder = self._G[retained_i, :] - G_split
        v_remainder = G_remainder / n_remainder
        raw_CP_remainder = self._CP[retained_i] - raw_CP_split
        centered_CP_remainder = self._nonnegative_or_error(
            raw_CP_remainder
            - n_remainder * np.inner(v_remainder, v_remainder),
            max(
                abs(self._CP[retained_i]),
                abs(raw_CP_split),
                abs(raw_CP_remainder),
            ),
            "cluster compactness",
        )
        raw_CP_remainder = (
            centered_CP_remainder
            + n_remainder * np.inner(v_remainder, v_remainder)
        )

        new_i = self._label_map.get_internal_label(new_label)
        if new_i != self._n_clusters:
            raise RuntimeError("New split label was not appended")

        self._n[retained_i] = n_remainder
        self._v[retained_i, :] = v_remainder
        self._CP[retained_i] = raw_CP_remainder
        self._G[retained_i, :] = G_remainder
        self._n.append(count)
        self._v = np.vstack((self._v, centroid))
        self._CP.append(raw_CP_split)
        self._G = np.vstack((self._G, G_split))
        self._n_clusters += 1
        self._rebuild_after_operation()

    def _rebuild_after_operation(self):
        """Rebuild the centroid-to-cluster dissimilarity matrix."""

        if self._n_clusters == 0:
            self._S = np.empty((0, 0))
            self._sil_coefs = []
            return

        counts = np.asarray(self._n)
        centroid_norms = np.einsum("ij,ij->i", self._v, self._v)
        values = (
            np.asarray(self._CP)[:, None]
            + counts[:, None] * centroid_norms[None, :]
            - 2 * (self._G @ self._v.T)
        ) / counts[:, None]
        self._S = np.fmax(values, 0.0)
        self._sil_coefs = np.zeros(self._n_clusters)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Centroid-based Silhouette (cSIL) CVI.
        """

        if self._n_clusters > 1:
            a = np.diag(self._S)
            other = self._S.copy()
            np.fill_diagonal(other, np.inf)
            b = np.min(other, axis=0)
            denominator = np.maximum(a, b)
            self._sil_coefs = np.divide(
                b - a,
                denominator,
                out=np.zeros(self._n_clusters),
                where=denominator != 0.0,
            )
            # cSIL index value
            self.criterion_value = np.sum(self._sil_coefs) / self._n_clusters

        else:
            self._sil_coefs = np.zeros(self._n_clusters)
            self.criterion_value = np.nan
