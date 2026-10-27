"""
Xie-Beni (XB) Cluster Validity Index.

References
----------
1. X. L. Xie and G. Beni, "A Validity Measure for Fuzzy Clustering," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 8, pp. 841-847, 1991.
2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base, _trial


# XB object definition
class XB(_base.CVI):
    """
    Xie-Beni (XB) Cluster Validity Index.

    References
    ----------
    1. X. L. Xie and G. Beni, "A Validity Measure for Fuzzy Clustering," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 8, pp. 841-847, 1991.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    info = _base.CVIInfo(
        name="Xie-Beni",
        name_short="XB",
        index_min=0.0,
        index_max=np.inf,
        optimality="min",
        backends=("numpy", "numba", "jax"),
    )
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy", capacity=None):
        """
        XB initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba", "jax"}, default="numpy"
            Select the numerical backend. Optional backends load on demand.
            JAX requires x64.
        capacity : int or None, default=None
            Opt into JAX streaming with this maximum number of clusters.
            Only available with backend="jax". No limit on sample count.
        """

        # Run the base initialization
        super().__init__(backend=backend, capacity=capacity)

        # XB-specific initialization
        self._mu = np.zeros([0])     # dim
        self._SEP = np.zeros([0])    # dim
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._WGSS = 0.0

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Xie-Beni (XB) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # XB-specific setup
        self._SEP = np.zeros([self._dim])
        self._mu = sample

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
        """Xie-Beni score of each plan slot."""
        within = self._trial_within(plan)
        separation = _trial.trial_min_off_diagonal(self._D, plan.rows, plan)
        k = self._trial_cluster_counts(plan)
        scores = within / ((self._n_samples + 1) * separation)
        scores[(k < 2) | ~(separation > 0.0)] = np.nan
        return scores

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Xie-Beni (XB) CVI.
        """

        self._setup_batch_statistics(data, labels)
        self._mu = np.mean(data, axis=0)
        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )

    def _rebuild_after_operation(self):
        """Rebuild centroid distances after a structural operation."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._SEP = np.zeros(0)
            self._D = np.zeros((0, 0))
            self._WGSS = 0.0
            return

        self._D = self._backend.pairwise_centroid_distances(
            self._v,
        )
        self._WGSS = sum(self._CP)
        if self._n_clusters < 2:
            self._SEP = 0.0

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Xie-Beni (XB) CVI.
        """

        if self._n_clusters > 1:
            # Within group sum of scatters
            self._WGSS = sum(self._CP)
            # # Between groups sum of scatters
            # self._BGSS = sum(self._SEP)
            self._SEP = self._backend.minimum_off_diagonal(self._D)
            if self._SEP > 0.0:
                # XB index value
                self.criterion_value = (
                    self._WGSS / (self._n_samples * self._SEP)
                )
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
