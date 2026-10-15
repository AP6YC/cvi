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
from . import _base, _batch


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
        backends=("numpy", "numba"),
    )

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
        self._centered_CP = []
        self._residuals = np.empty((0, 0))

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Centroid-based Silhouette (cSIL) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

    def _ensure_centered_state(self):
        """Recover available centered state from older serialized objects."""
        if not len(self._CP):
            self._centered_CP = []
            self._residuals = np.empty((0, self._dim))
            return
        if len(getattr(self, "_centered_CP", [])) == len(self._CP):
            if hasattr(self, "_residuals"):
                return
        counts = np.asarray(self._n[:len(self._CP)])
        self._centered_CP = (counts * np.diag(self._S)).tolist()
        self._residuals = self._G - counts[:, None] * self._v[:len(counts)]

    def _sync_raw_moments(self):
        """Retain the legacy raw fields; all evaluation uses centered state."""
        counts = np.asarray(self._n)
        self._G = counts[:, None] * self._v + self._residuals
        self._CP = (np.asarray(self._centered_CP)
                    + counts * np.sum(self._v ** 2, axis=1)
                    + 2 * np.sum(self._v * self._residuals, axis=1)).tolist()

    def _merge_batch_statistics(self, data, order, offsets, slots):
        """Merge centered moments into a staged mini-batch candidate."""
        self._ensure_centered_state()
        compactness, self._residuals = _batch.merge_moments(
            self, data, order, offsets, slots, self._centered_CP, self._residuals,
        )
        self._centered_CP = compactness.tolist()

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """Update centered moments and just the affected dissimilarity row/column."""
        self._ensure_centered_state()
        sample = np.asarray(sample, dtype=np.float64)
        i = self._label_map.get_internal_label(label)
        if self._n_samples == 0:
            self._setup(sample)
            self._residuals = np.empty((0, self._dim))
        if i == self._n_clusters:
            self._n.append(1)
            self._v = np.vstack((self._v, sample))
            self._centered_CP.append(0.0)
            self._residuals = np.vstack((self._residuals, np.zeros(self._dim)))
            self._CP.append(np.inner(sample, sample))
            self._G = np.vstack((self._G, sample))
            matrix = np.zeros((i + 1, i + 1))
            matrix[:i, :i] = self._S
            self._S = matrix
            self._n_clusters += 1
        else:
            n = self._n[i]
            center = self._v[i] + (sample - self._v[i]) / (n + 1)
            shift = self._v[i] - center
            difference = sample - center
            self._centered_CP[i] += (
                np.dot(difference, difference) + n * np.dot(shift, shift)
                + 2 * np.dot(shift, self._residuals[i])
            )
            self._residuals[i] += difference + n * shift
            self._n[i], self._v[i] = n + 1, center
            self._CP[i] += np.inner(sample, sample)
            self._G[i] += sample
        self._n_samples += 1
        differences = self._v[i] - self._v
        distances = np.einsum("ij,ij->i", differences, differences)
        self._S[i, :] = distances + (
            self._centered_CP[i] + 2 * (differences @ self._residuals[i])
        ) / self._n[i]
        self._S[:, i] = distances + (
            np.asarray(self._centered_CP)
            - 2 * np.einsum("ij,ij->i", differences, self._residuals)
        ) / np.asarray(self._n)

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Centroid-based Silhouette (cSIL) CVI.
        """

        order, offsets = self._setup_batch_statistics(data, labels)
        self._centered_CP = list(self._CP)
        self._CP, self._G, self._S, self._residuals = self._backend.silhouette_batch_statistics(
            data, order, offsets, self._v, self._CP,
        )

    def _delete_cluster(self, label: int, i_label: int):
        """Delete one cSIL cluster and compact its internal label."""

        self._centered_CP = self._delete_vector_entry(self._centered_CP, i_label)
        self._residuals = np.delete(self._residuals, i_label, axis=0)
        self._CP = self._delete_vector_entry(self._CP, i_label)
        self._G = np.delete(self._G, i_label, axis=0)
        super()._delete_cluster(label, i_label)

    def _remove(self, sample: np.ndarray, label: int, i_label: int):
        """Subtract a sample using centered moments, without raw cancellation."""
        self._ensure_centered_state()
        n_old = self._n[i_label]
        center = self._v[i_label].copy()
        if n_old == 1:
            self._validate_singleton_removal(sample, center)
            self._delete_cluster(label, i_label)
            self._n_samples -= 1
            if self._n_samples == 0:
                self._clear_common_state()
            self._rebuild_after_operation()
            return
        n_new = n_old - 1
        difference = sample - center
        remaining_residual = self._residuals[i_label] - difference
        new_center = center + remaining_residual / n_new
        shift = center - new_center
        q = self._centered_CP[i_label]
        new_q = (q - np.dot(difference, difference)
                 + n_new * np.dot(shift, shift)
                 + 2 * np.dot(shift, remaining_residual))
        new_q = self._nonnegative_or_error(new_q, q, "cluster compactness")
        if n_new == 1:
            new_q = 0.0
        self._n[i_label], self._v[i_label] = n_new, new_center
        self._centered_CP[i_label] = new_q
        self._residuals[i_label] = remaining_residual + n_new * shift
        if n_new == 1:
            self._residuals[i_label] = 0.0
        self._n_samples -= 1
        self._rebuild_after_operation()

    def _merge(self, target_label, source_label, target_i, source_i):
        """Merge two centered cluster summaries."""
        self._ensure_centered_state()
        n, v, q, r = self._backend.merge_statistics(
            np.array([self._n[target_i]]), self._v[[target_i]],
            np.array([self._centered_CP[target_i]]), self._residuals[[target_i]],
            np.array([self._n[source_i]]), self._v[[source_i]],
            np.array([self._centered_CP[source_i]]), self._residuals[[source_i]],
        )
        self._n[target_i], self._v[target_i] = int(n[0]), v[0]
        self._centered_CP[target_i], self._residuals[target_i] = float(q[0]), r[0]
        self._delete_cluster(source_label, source_i)
        self._rebuild_after_operation()

    def _split(self, new_label, retained_i, count, centroid, compactness, covariance):
        """Subtract a centered subset summary from its parent cluster."""
        if compactness is None:
            raise ValueError("cSIL split requires compactness")
        self._ensure_centered_state()
        n = self._n[retained_i] - count
        center = self._v[retained_i].copy()
        difference = centroid - center
        residual = self._residuals[retained_i] - count * difference
        remaining_center = center + residual / n
        shift = center - remaining_center
        parent_q = self._centered_CP[retained_i]
        remaining_q = (parent_q - compactness - count * np.dot(difference, difference)
                       + n * np.dot(shift, shift) + 2 * np.dot(shift, residual))
        remaining_q = self._nonnegative_or_error(
            remaining_q, max(parent_q, compactness), "cluster compactness",
        )
        new_i = self._label_map.get_internal_label(new_label)
        if new_i != self._n_clusters:
            raise RuntimeError("New split label was not appended")
        self._n[retained_i], self._v[retained_i] = n, remaining_center
        self._centered_CP[retained_i] = remaining_q
        self._residuals[retained_i] = residual + n * shift
        self._n.append(count)
        self._v = np.vstack((self._v, centroid))
        self._centered_CP.append(compactness)
        self._residuals = np.vstack((self._residuals, np.zeros(self._dim)))
        self._n_clusters += 1
        self._rebuild_after_operation()

    def _rebuild_after_operation(self):
        """Rebuild dissimilarities directly from centered moments."""
        if self._n_clusters == 0:
            self._S = np.empty((0, 0))
            self._sil_coefs = []
            self._centered_CP = []
            self._residuals = np.empty((0, self._dim))
            return
        self._sync_raw_moments()
        self._S = self._backend.silhouette_from_moments(
            self._v, np.asarray(self._centered_CP), self._residuals, np.asarray(self._n),
        )
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
