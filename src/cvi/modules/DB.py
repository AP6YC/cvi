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
from . import _base


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
        optimality="min"
    )
    _supports_numba = True
    _supports_remove_merge = True
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

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Davies-Bouldin (DB) CVI.
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
            S_new = 0.0
            if self._n_clusters == 0:
                D_new = np.zeros((1, 1))
            else:
                D_new = np.zeros((self._n_clusters + 1, self._n_clusters + 1))
                D_new[0:self._n_clusters, 0:self._n_clusters] = self._D
                d_column_new = np.zeros(self._n_clusters + 1)
                d_column_new[:-1] = self._backend.centroid_distances(
                    self._v, v_new,
                )
                D_new[i_label, :] = d_column_new
                D_new[:, i_label] = d_column_new

            # Update 1-D parameters with list appends
            self._n_clusters += 1
            self._n.append(n_new)
            self._CP.append(CP_new)
            self._S.append(S_new)

            # Update 2-D parameters with numpy vstacks
            self._v = np.vstack([self._v, v_new])
            self._G = np.vstack([self._G, G_new])
            self._D = D_new

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
            S_new = CP_new / n_new
            d_column_new = self._backend.centroid_distances(
                self._v, v_new,
            )
            d_column_new[i_label] = 0.0

            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._CP[i_label] = CP_new
            self._G[i_label, :] = G_new
            self._S[i_label] = S_new
            self._D[i_label, :] = d_column_new
            # self._D[:, i_label] = np.tranpose(d_column_new)
            self._D[:, i_label] = d_column_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new

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
            separations = self._D[
                np.triu_indices(self._n_clusters, k=1)
            ]
            if np.all(separations > 0.0):
                for ix in range(self._n_clusters - 1):
                    for jx in range(ix + 1, self._n_clusters):
                        self._R[jx, ix] = (
                            (self._S[ix] + self._S[jx]) / self._D[jx, ix]
                        )
                self._R = self._R + np.transpose(self._R)
                self.criterion_value = (
                    np.sum(np.max(self._R, axis=0)) / self._n_clusters
                )
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
