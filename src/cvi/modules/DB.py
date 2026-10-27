"""
Davies-Bouldin (DB) Cluster Validity Index.

References
----------
1. D. L. Davies and D. W. Bouldin, "A cluster separation measure," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol. 1, no. 2, pp. 224-227, Feb. 1979.
2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base, _trial


# DB object definition
class DB(_base.CVI):
    """
    Davies-Bouldin (DB) Cluster Validity Index.

    References
    ----------
    1. D. L. Davies and D. W. Bouldin, "A cluster separation measure," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol. 1, no. 2, pp. 224-227, Feb. 1979.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    info = _base.CVIInfo(
        name="Davies-Bouldin",
        name_short="DB",
        index_min=0.0,
        index_max=np.inf,
        optimality="min",
        backends=("numpy", "numba"),
    )
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy"):
        """
        Davies-Bouldin (DB) initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

        # CH-specific initialization
        self._mu = np.zeros([0])     # dim
        self._R = np.zeros([0, 0])   # n_clusters x dim
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._S = []                 # dim

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Davies-Bouldin (DB) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # DB-specific setup
        self._mu = sample

    def _plan_add_index(self, plan):
        """Prospective scatter and squared centroid distances."""
        plan.S_new = plan.CP_new / plan.n_new
        plan.rows = self._candidate_rows(
            plan,
            lambda slot: self._backend.centroid_distances(
                self._v, plan.v_new[slot],
            ),
        )

    def _apply_add_index(self, plan, slot, i_label, new):
        """Write the planned scatter and distance row."""
        if plan.fresh:
            self._S.append(0.0)
            self._D = np.zeros((1, 1))
            return
        if new:
            self._S.append(0.0)
        else:
            self._S[i_label] = float(plan.S_new[slot])
        self._D = self._apply_pairwise_row(self._D, plan.rows[slot], i_label, new)

    def _score_add(self, plan):
        """Davies-Bouldin score of each plan slot.

        Only one cluster's row and column of R change, so each row's maximum
        is its largest entry outside the changed column, or the new entry.
        """
        existing, cand = plan.existing, plan.cand
        slots = np.flatnonzero(existing)
        scatter = np.asarray(self._S, dtype=np.float64)
        ratios = (scatter[:, None] + scatter[None, :]) / self._D
        others = _trial.extremes_without(
            _trial.row_extremes(ratios, largest=True),
            np.where(existing, cand, -1),
        )
        cross = (scatter[None, :] + plan.S_new[:, None]) / plan.rows
        maxima = np.maximum(others, cross)
        own = np.max(_trial.masked_self(cross, plan, -np.inf), axis=1)
        maxima[slots, cand[existing]] = own[existing]
        k = self._trial_cluster_counts(plan)
        scores = (np.sum(maxima, axis=1) + np.where(existing, 0.0, own)) / k
        separation = _trial.trial_min_off_diagonal(self._D, plan.rows, plan)
        scores[(k < 2) | ~(separation > 0.0)] = np.nan
        return scores

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Davies-Bouldin (DB) CVI.
        """

        self._setup_batch_statistics(data, labels)
        self._mu = np.mean(data, axis=0)
        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )
        self._S = [cp / n for cp, n in zip(self._CP, self._n)]

    def _rebuild_after_operation(self):
        """Rebuild dispersion and centroid-distance state."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._S = []
            self._D = np.zeros((0, 0))
            self._R = np.zeros((0, 0))
            return

        self._S = [
            self._CP[ix] / self._n[ix]
            for ix in range(self._n_clusters)
        ]
        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )
        self._R = np.zeros((self._n_clusters, self._n_clusters))

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Davies-Bouldin (DB) CVI.
        """

        if self._n_clusters > 1:
            self._R = np.zeros((self._n_clusters, self._n_clusters))
            rows, columns = np.triu_indices(self._n_clusters, k=1)
            separations = self._D[rows, columns]
            if np.all(separations > 0.0):
                # Evaluate only distinct pairs: the diagonal stays exactly zero,
                # without computing unused self-ratios or overflowing S_i + S_i.
                scatter = np.asarray(self._S)
                ratios = ((scatter[rows] + scatter[columns])
                          / self._D[columns, rows])
                self._R[columns, rows] = ratios
                self._R[rows, columns] = ratios
                self.criterion_value = (
                    np.sum(np.max(self._R, axis=0)) / self._n_clusters
                )
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
