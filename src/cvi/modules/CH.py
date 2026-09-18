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
        optimality="max"
    )
    _supports_numba = True
    _supports_remove_merge = True
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy"):
        """
        CH initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

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

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Calinski-Harabasz (CH) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self._label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self._n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self._n_samples == 0:
            self._setup(sample)
        else:
            self._mu = (
                (1 - 1/n_samples_new) * self._mu
                + (1/n_samples_new) * sample
            )

        # IF NEW CLUSTER LABEL
        # Correct for python 0-indexing
        if i_label > self._n_clusters - 1:
            n_new = 1
            v_new = sample
            CP_new = 0.0
            G_new = np.zeros(self._dim)

            # Update 1-D parameters with list appends
            self._n_clusters += 1
            self._n.append(n_new)
            self._CP.append(CP_new)

            # Update 2-D parameters with numpy vstacks
            self._v = np.vstack([self._v, v_new])
            self._G = np.vstack([self._G, G_new])

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self._n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self._v[i_label, :]
                + (1 / n_new) * sample
            )
            delta_v = self._v[i_label, :] - v_new
            diff_x_v = sample - v_new
            CP_new = (
                self._CP[i_label]
                + np.inner(diff_x_v, diff_x_v)
                + self._n[i_label] * np.inner(delta_v, delta_v)
                + 2 * np.inner(delta_v, self._G[i_label, :])
            )
            G_new = (
                self._G[i_label, :]
                + diff_x_v
                + self._n[i_label] * delta_v
            )
            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._CP[i_label] = CP_new
            self._G[i_label, :] = G_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new
        # self._mu = mu_new
        self._SEP = np.array([
            self._n[ix] * sum((self._v[ix, :] - self._mu) ** 2)
            for ix in range(self._n_clusters)
        ])

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
        """Rebuild separation statistics after a remove or merge."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._SEP = np.zeros(0)
            self._BGSS = 0.0
            self._WGSS = 0.0
            return

        self._SEP = np.asarray([
            self._n[ix] * np.sum((self._v[ix, :] - self._mu) ** 2)
            for ix in range(self._n_clusters)
        ])
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
            # CH index value
            self.criterion_value = (
                (self._BGSS / self._WGSS)
                * ((self._n_samples - self._n_clusters) / (self._n_clusters - 1))
            )
        else:
            self._BGSS = 0.0
            self.criterion_value = 0.0
