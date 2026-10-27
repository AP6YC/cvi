"""
Calinski-Harabasz (CH) Cluster Validity Index.

References
----------
1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
2. T. Calinski and J. Harabasz, "A dendrite method for cluster analysis," Communications in Statistics, vol. 3, no. 1, pp. 1-27, 1974.
3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
4. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# CH object definition
class CH(_base.CVI):
    """
    Calinski-Harabasz (CH) Cluster Validity Index.

    References
    ----------
    1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    2. T. Calinski and J. Harabasz, "A dendrite method for cluster analysis," Communications in Statistics, vol. 3, no. 1, pp. 1-27, 1974.
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    4. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """
    info = _base.CVIInfo(
        name="Calinski-Harabasz",
        name_short="CH",
        index_min=0.0,
        index_max=np.inf,
        optimality="max",
        backends=("numpy", "numba", "jax"),
    )
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy", capacity=None):
        """
        CH initialization routine.

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

        # CH-specific initialization
        self._mu = np.zeros([0])     # dim
        self._SEP = np.zeros([0])     # dim
        self._BGSS = 0.0
        self._WGSS = 0.0

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Calinski-Harabasz (CH) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # CH-specific setup
        self._SEP = np.zeros([self._dim])
        self._mu = sample

    def _plan_add_index(self, plan):
        """Separation terms against the prospective global mean."""
        plan.sep_base = np.asarray(self._n) * self._backend.centroid_distances(
            self._v, plan.mu_new,
        )
        plan.sep_new = plan.n_new * self._backend.centroid_distances(
            plan.v_new, plan.mu_new,
        )

    def _trial_between(self, plan):
        """Between-group sum of squares for each plan slot."""
        total = np.sum(plan.sep_base)
        between = total + plan.sep_new
        existing = plan.existing
        between[existing] = (
            total - plan.sep_base[plan.cand[existing]] + plan.sep_new[existing]
        )
        return between

    def _apply_add_index(self, plan, slot, i_label, new):
        """Update separation terms from the plan."""
        if plan.fresh:
            self._SEP = np.asarray(self._n) * self._backend.centroid_distances(
                self._v, self._mu,
            )
            return
        separation = plan.sep_base.copy()
        if new:
            separation = np.append(separation, plan.sep_new[slot])
        else:
            separation[i_label] = plan.sep_new[slot]
        self._SEP = separation

    def _score_add(self, plan):
        """Calinski-Harabasz score of each plan slot."""
        within = self._trial_within(plan)
        between = self._trial_between(plan)
        k = self._trial_cluster_counts(plan)
        n = self._n_samples + 1
        scores = (between / within) * ((n - k) / (k - 1))
        scores[(k < 2) | ~(within > 0.0)] = np.nan
        return scores

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Calinski-Harabasz (CH) CVI.
        """

        self._setup_batch_statistics(data, labels)
        self._mu = np.mean(data, axis=0)
        self._SEP = np.asarray(self._n) * self._backend.centroid_distances(
            self._v, self._mu,
        )

    def _rebuild_after_operation(self):
        """Rebuild separation statistics after a structural operation."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._SEP = np.zeros(0)
            self._BGSS = 0.0
            self._WGSS = 0.0
            return

        self._SEP = np.asarray(self._n) * self._backend.centroid_distances(
            self._v, self._mu,
        )
        self._WGSS = sum(self._CP)
        self._BGSS = sum(self._SEP)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Calinski-Harabasz (CH) CVI.
        """

        if self._n_clusters > 1:
            # Within group sum of scatters
            self._WGSS = sum(self._CP)
            # Between groups sum of scatters
            self._BGSS = sum(self._SEP)
            if self._WGSS > 0.0:
                # CH index value
                self.criterion_value = (
                    (self._BGSS / self._WGSS)
                    * (
                        (self._n_samples - self._n_clusters)
                        / (self._n_clusters - 1)
                    )
                )
            else:
                self.criterion_value = np.nan
        else:
            self._BGSS = 0.0
            self.criterion_value = np.nan
