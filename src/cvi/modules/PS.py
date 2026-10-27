"""
Partition Separation (PS) Cluster Validity Index.

References
----------
1. Miin-Shen Yang and Kuo-Lung Wu, "A new validity index for fuzzy clustering," 10th IEEE International Conference on Fuzzy Systems. (Cat. No.01CH37297), Melbourne, Victoria, Australia, 2001, pp. 89-92, vol.1.
2. E. Lughofer, "Extensions of vector quantization for incremental clustering," Pattern Recognit., vol. 41, no. 3, pp. 995-1011, 2008.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base, _trial


# PS object definition
class PS(_base.CVI):
    """
    Partition Separation (PS) Cluster Validity Index.

    References
    ----------
    1. Miin-Shen Yang and Kuo-Lung Wu, "A new validity index for fuzzy clustering," 10th IEEE International Conference on Fuzzy Systems. (Cat. No.01CH37297), Melbourne, Victoria, Australia, 2001, pp. 89-92, vol.1.
    2. E. Lughofer, "Extensions of vector quantization for incremental clustering," Pattern Recognit., vol. 41, no. 3, pp. 995-1011, 2008.
    """

    info = _base.CVIInfo(
        name="Partition Separation",
        name_short="PS",
        index_min=0.0,
        index_max=1.0,
        optimality="max",
        backends=("numpy", "numba"),
    )

    def __init__(self, *, backend="numpy"):
        """
        Partition Separation (PS) initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

        # PS-specific initialization
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._v_bar = []
        self._beta_t = 0.0
        self._PS_i = np.zeros(0)

    def _plan_add_index(self, plan):
        """Squared distances from each prospective centroid to the others."""
        plan.rows = self._candidate_rows(
            plan,
            lambda slot: self._backend.centroid_distances(
                self._v, plan.v_new[slot],
            ),
        )

    def _apply_add_index(self, plan, slot, i_label, new):
        """Write the planned distance row."""
        if plan.fresh:
            self._D = np.zeros((1, 1))
            return
        self._D = self._apply_pairwise_row(self._D, plan.rows[slot], i_label, new)

    def _score_add(self, plan):
        """Partition Separation score of each plan slot."""
        existing, cand = plan.existing, plan.cand
        slots = np.flatnonzero(existing)
        k_old = self._n_clusters
        k = self._trial_cluster_counts(plan)

        counts = np.asarray(self._n, dtype=np.float64)
        largest = np.maximum(np.max(counts), plan.n_new)
        count_terms = (np.sum(counts) + 1) / largest

        nearest = np.minimum(
            _trial.extremes_without(
                _trial.row_extremes(self._D, largest=False),
                np.where(existing, cand, -1),
            ),
            plan.rows,
        )
        own = np.min(_trial.masked_self(plan.rows, plan, np.inf), axis=1)
        nearest[slots, cand[existing]] = own[existing]

        mean = np.mean(self._v, axis=0)
        beta = np.empty(plan.n_slots)
        for slot in range(plan.n_slots):
            v_new = plan.v_new[slot]
            if existing[slot]:
                i = cand[slot]
                mean_new = mean + (v_new - self._v[i]) / k_old
                spread = self._backend.centroid_distances(self._v, mean_new)
                spread[i] = np.sum((v_new - mean_new) ** 2)
                beta[slot] = np.mean(spread)
            else:
                mean_new = (k_old * mean + v_new) / (k_old + 1)
                spread = self._backend.centroid_distances(self._v, mean_new)
                beta[slot] = (
                    np.sum(spread) + np.sum((v_new - mean_new) ** 2)
                ) / (k_old + 1)

        separation = (
            np.sum(np.exp(-nearest / beta[:, None]), axis=1)
            + np.where(existing, 0.0, np.exp(-own / beta))
        )
        scores = count_terms - separation
        scores[(k < 2) | ~(beta > 0.0)] = np.nan
        return scores

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Partition Separation (PS) CVI.
        """

        self._setup_batch_statistics(data, labels, compactness=False)
        # Keep the unused common fields consistent with incremental setup.
        self._CP = []
        self._G = np.zeros((0, self._dim))
        self._mu = np.mean(data, axis=0)
        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )

    def _remove(self, sample: np.ndarray, label: int, i_label: int):
        """Remove one sample from the PS centroid statistics."""

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
        v_new = (n_old * v_old - sample) / n_new

        self._n[i_label] = n_new
        self._v[i_label, :] = v_new
        self._n_samples = n_samples_new
        self._rebuild_after_operation()

    def _merge(
        self,
        target_label: int,
        source_label: int,
        target_i: int,
        source_i: int,
    ):
        """Merge two PS centroid summaries."""

        n_target = self._n[target_i]
        n_source = self._n[source_i]
        n_new = n_target + n_source
        v_new = (
            n_target * self._v[target_i, :]
            + n_source * self._v[source_i, :]
        ) / n_new

        self._n[target_i] = n_new
        self._v[target_i, :] = v_new
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
        """Split count and centroid statistics from a PS cluster."""

        n_parent = self._n[retained_i]
        n_remainder = n_parent - count
        v_parent = self._v[retained_i, :].copy()
        v_remainder = (
            n_parent * v_parent - count * centroid
        ) / n_remainder

        new_i = self._label_map.get_internal_label(new_label)
        if new_i != self._n_clusters:
            raise RuntimeError("New split label was not appended")

        self._n[retained_i] = n_remainder
        self._v[retained_i, :] = v_remainder
        self._n.append(count)
        self._v = np.vstack((self._v, centroid))
        self._n_clusters += 1
        self._rebuild_after_operation()

    def _rebuild_after_operation(self):
        """Rebuild pairwise centroid distances."""

        if self._n_clusters == 0:
            self._D = np.zeros((0, 0))
            self._v_bar = []
            self._beta_t = 0.0
            self._PS_i = np.zeros(0)
            return

        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )
        self._v_bar = []
        self._beta_t = 0.0
        self._PS_i = np.zeros(self._n_clusters)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Partition Separation (PS) CVI.
        """

        if self._n_clusters > 1:
            self._v_bar = np.mean(self._v, axis=0)
            self._beta_t = np.mean(self._backend.centroid_distances(
                self._v, self._v_bar,
            ))
            if self._beta_t > 0.0:
                counts = np.asarray(self._n)
                nearest = self._D.copy()
                np.fill_diagonal(nearest, np.inf)
                self._PS_i = (
                    counts / np.max(counts)
                    - np.exp(-np.min(nearest, axis=1) / self._beta_t)
                )
                self.criterion_value = np.sum(self._PS_i)
            else:
                self._PS_i = np.zeros(self._n_clusters)
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
