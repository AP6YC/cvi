"""
Generalized Dunn's Index 53 (GD53) Cluster Validity Index.

References
----------
1. A. Ibrahim, J. M. Keller, and J. C. Bezdek, "Evaluating Evolving Structure in Streaming Data With Modified Dunn's Indices," IEEE Transactions on Emerging Topics in Computational Intelligence, pp. 1-12, 2019.
2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML].
3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
4. J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters," J. Cybern., vol. 3, no. 3 , pp. 32-57, 1973.
5. J. C. Bezdek and N. R. Pal, "Some new indexes of cluster validity," IEEE Trans. Syst., Man, and Cybern., vol. 28, no. 3, pp. 301-315, Jun. 1998.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base, _trial


# GD53 object definition
class GD53(_base.CVI):
    """
    Generalized Dunn's Index 53 (GD53) Cluster Validity Index.

    References
    ----------
    1. A. Ibrahim, J. M. Keller, and J. C. Bezdek, "Evaluating Evolving Structure in Streaming Data With Modified Dunn's Indices," IEEE Transactions on Emerging Topics in Computational Intelligence, pp. 1-12, 2019.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    4. J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters," J. Cybern., vol. 3, no. 3 , pp. 32-57, 1973.
    5. J. C. Bezdek and N. R. Pal, "Some new indexes of cluster validity," IEEE Trans. Syst., Man, and Cybern., vol. 28, no. 3, pp. 301-315, Jun. 1998.
    """

    info = _base.CVIInfo(
        name="Generalized Dunn's 53",
        name_short="GD53",
        index_min=0.0,
        index_max=np.inf,
        optimality="max",
        backends=("numpy", "numba"),
    )
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy"):
        """
        Generalized Dunn's Index 53 (GD53) initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

        # GD53-specific initialization
        self._mu = np.zeros([0])     # dim
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._inter = 0.0
        self._intra = 0.0

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Generalized Dunn's Index 53 (GD53) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # GD53-specific setup
        self._mu = sample

    @staticmethod
    def _dispersion_row(compactness, counts, compactness_new, count_new):
        """Return GD53 dispersion from one cluster to every cluster."""
        return ((compactness_new + np.asarray(compactness))
                / (count_new + np.asarray(counts)))

    @classmethod
    def _dispersion_matrix(cls, compactness, counts):
        """Return the symmetric GD53 dispersion matrix with a zero diagonal."""
        compactness = np.asarray(compactness)
        counts = np.asarray(counts)
        matrix = ((compactness[:, None] + compactness[None, :])
                  / (counts[:, None] + counts[None, :]))
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _plan_add_index(self, plan):
        """Dispersion between each prospective cluster and the others."""
        compactness = np.asarray(self._CP, dtype=np.float64)
        counts = np.asarray(self._n)
        plan.rows = self._candidate_rows(
            plan,
            lambda slot: (
                (plan.CP_new[slot] + compactness)
                / (plan.n_new[slot] + counts)
            ),
        )

    def _apply_add_index(self, plan, slot, i_label, new):
        """Write the planned dispersion row."""
        if plan.fresh:
            self._D = np.zeros((1, 1))
            return
        self._D = self._apply_pairwise_row(self._D, plan.rows[slot], i_label, new)

    def _score_add(self, plan):
        """Generalized Dunn score of each plan slot."""
        intra = 2 * _trial.max_without(
            np.divide(self._CP, self._n), plan, plan.CP_new / plan.n_new,
        )
        inter = _trial.trial_min_off_diagonal(self._D, plan.rows, plan)
        k = self._trial_cluster_counts(plan)
        scores = inter / intra
        scores[(k < 2) | ~(intra > 0.0)] = np.nan
        return scores

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Generalized Dunn's Index 53 (GD53) CVI.
        """

        self._setup_batch_statistics(data, labels)
        self._mu = np.mean(data, axis=0)
        self._D = self._dispersion_matrix(self._CP, self._n)

    def _rebuild_after_operation(self):
        """Rebuild pairwise dispersion after a structural operation."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._D = np.zeros((0, 0))
            self._inter = 0.0
            self._intra = 0.0
            return

        self._D = self._dispersion_matrix(self._CP, self._n)
        if self._n_clusters < 2:
            self._inter = 0.0
            self._intra = 0.0

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Generalized Dunn's Index 53 (GD53) CVI.
        """

        if self._n_clusters > 1:
            self._intra = 2 * np.max(np.divide(self._CP, self._n))
            # Between-group measure of separation/isolation
            self._inter = self._backend.minimum_off_diagonal(self._D)
            # GD53 index value
            if self._intra > 0.0:
                self.criterion_value = self._inter / self._intra
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
