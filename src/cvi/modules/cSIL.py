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
from . import _base, _trial, _batch


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

    def _plan_add_statistics(self, plan):
        """Centered and raw moments for every plan slot."""
        self._ensure_centered_state()
        self._plan_centroids(plan, shift_form=True)
        x, cand, existing = plan.x, plan.cand, plan.existing
        squared_norm = np.inner(x, x)
        if plan.n_slots == 1:
            self._plan_single_moments(plan, squared_norm)
            return
        plan.Q_new = np.zeros(plan.n_slots)
        plan.r_new = np.zeros((plan.n_slots, len(x)))
        plan.raw_CP_new = np.full(plan.n_slots, squared_norm)
        plan.raw_G_new = np.tile(x, (plan.n_slots, 1))
        if existing.any():
            i = cand[existing]
            n = plan.n_old[existing]
            center = plan.v_new[existing]
            shift = self._v[i] - center
            difference = x - center
            residuals = self._residuals[i]
            plan.Q_new[existing] = np.asarray(self._centered_CP)[i] + (
                _trial.row_dot(difference, difference)
                + n * _trial.row_dot(shift, shift)
                + 2 * _trial.row_dot(shift, residuals)
            )
            plan.r_new[existing] = residuals + (difference + n[:, None] * shift)
            plan.raw_CP_new[existing] = (
                np.asarray(self._CP, dtype=np.float64)[i] + squared_norm
            )
            plan.raw_G_new[existing] = self._G[i] + x

    def _plan_single_moments(self, plan, squared_norm):
        """One-slot form of the moment update, using plain 1-D arithmetic."""
        x = plan.x
        if plan.existing[0]:
            i = int(plan.cand[0])
            n = plan.n_old[0]
            center = plan.v_new[0]
            shift = self._v[i] - center
            difference = x - center
            residuals = self._residuals[i]
            Q_new = self._centered_CP[i] + (
                _trial.dot(difference, difference)
                + n * _trial.dot(shift, shift)
                + 2 * _trial.dot(shift, residuals)
            )
            r_new = residuals + (difference + n * shift)
            raw_CP_new = self._CP[i] + squared_norm
            raw_G_new = self._G[i] + x
        else:
            Q_new, r_new = 0.0, np.zeros(len(x))
            raw_CP_new, raw_G_new = squared_norm, x
        plan.Q_new = np.asarray([Q_new], dtype=np.float64)
        plan.r_new = r_new.reshape(1, -1).astype(np.float64, copy=True)
        plan.raw_CP_new = np.asarray([raw_CP_new], dtype=np.float64)
        plan.raw_G_new = raw_G_new.reshape(1, -1).astype(np.float64, copy=True)

    def _plan_add_index(self, plan):
        """Prospective dissimilarity row and column of each slot."""
        compactness = np.asarray(self._centered_CP)
        counts = np.asarray(self._n)
        plan.rows = np.empty((plan.n_slots, self._n_clusters))
        plan.cols = np.empty((plan.n_slots, self._n_clusters))
        for slot in range(plan.n_slots):
            differences = plan.v_new[slot] - self._v
            distances = np.einsum("ij,ij->i", differences, differences)
            plan.rows[slot] = distances + (
                plan.Q_new[slot] + 2 * (differences @ plan.r_new[slot])
            ) / plan.n_new[slot]
            plan.cols[slot] = distances + (
                compactness
                - 2 * np.einsum("ij,ij->i", differences, self._residuals)
            ) / counts
        existing = plan.existing
        own = plan.Q_new / plan.n_new
        slots = np.flatnonzero(existing)
        plan.rows[slots, plan.cand[existing]] = own[existing]
        plan.cols[slots, plan.cand[existing]] = own[existing]

    def _apply_add_index(self, plan, slot, i_label, new):
        """Write planned moments and the dissimilarity row and column."""
        if plan.fresh:
            self._centered_CP = []
            self._residuals = np.empty((0, self._dim))
        if new:
            self._centered_CP.append(float(plan.Q_new[slot]))
            self._residuals = np.vstack((self._residuals, plan.r_new[slot]))
            self._CP.append(float(plan.raw_CP_new[slot]))
            self._G = np.vstack((self._G, plan.raw_G_new[slot]))
            k = self._S.shape[0]
            matrix = np.zeros((k + 1, k + 1))
            matrix[:k, :k] = self._S
            if not plan.fresh:
                matrix[k, :k] = plan.rows[slot]
                matrix[:k, k] = plan.cols[slot]
            matrix[k, k] = plan.Q_new[slot] / plan.n_new[slot]
            self._S = matrix
        else:
            self._centered_CP[i_label] = float(plan.Q_new[slot])
            self._residuals[i_label] = plan.r_new[slot]
            self._CP[i_label] = float(plan.raw_CP_new[slot])
            self._G[i_label] = plan.raw_G_new[slot]
            self._S[i_label, :] = plan.rows[slot]
            self._S[:, i_label] = plan.cols[slot]

    @staticmethod
    def _silhouettes(within, between):
        """Silhouette coefficients, zero where both terms vanish."""
        denominator = np.maximum(within, between)
        return np.divide(
            between - within,
            denominator,
            out=np.zeros(np.broadcast(within, between).shape),
            where=denominator != 0.0,
        )

    def _score_add(self, plan):
        """Centroid-based silhouette score of each plan slot.

        Adding to cluster i changes row i (its samples) and column i (its
        centroid) of S; every other column's minimum is updated from its
        best entry outside row i.
        """
        existing, cand = plan.existing, plan.cand
        slots = np.flatnonzero(existing)
        within = np.tile(np.diag(self._S), (plan.n_slots, 1))
        between = np.minimum(
            _trial.extremes_without(
                _trial.row_extremes(self._S.T, largest=False),
                np.where(existing, cand, -1),
            ),
            plan.rows,
        )
        own_within = plan.Q_new / plan.n_new
        own_between = np.min(
            _trial.masked_self(plan.cols, plan, np.inf), axis=1,
        )
        within[slots, cand[existing]] = own_within[existing]
        between[slots, cand[existing]] = own_between[existing]
        totals = (
            np.sum(self._silhouettes(within, between), axis=1)
            + np.where(
                existing, 0.0, self._silhouettes(own_within, own_between),
            )
        )
        k = self._trial_cluster_counts(plan)
        scores = totals / k
        scores[k < 2] = np.nan
        return scores

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
