"""
The (Renyi's) representative Cross Information Potential (rCIP) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# rCIP object definition
class rCIP(_base.CVI):
    """
    The stateful information of the (Renyi's) representative Cross Information Potential (rCIP) Cluster Validity Index.

    References
    ----------
    1. D. L. Davies and D. W. Bouldin, "A cluster separation measure," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol. 1, no. 2, pp. 224-227, Feb. 1979.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    def __init__(self):
        """
        (Renyi's) representative Cross Information Potential (rCIP) initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # rCIP-specific initialization
        self._D = np.zeros([0, 0])          # n_clusters x n_clusters
        self._sigmal = np.zeros([0, 0, 0])  # dim x dim x n_clusters
        self._delta_term = np.zeros([0, 0]) # dim x dim
        self._constant = 0.0
        # NO USE FOR CP IN rCIP
        self._CP = None

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        (Renyi's) representative Cross Information Potential (rCIP) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # rCIP-specific setup
        self._sigma = np.zeros([self._dim, self._dim, 0])
        epsilon = 12.0
        delta = 10.0 ** (-epsilon / self._dim)
        self._delta_term = np.eye(self._dim) * delta

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the (Renyi's) representative Cross Information Potential (rCIP) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self._label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self._n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self._n_samples == 0:
            self._setup(sample)

        self._constant = 1 / np.sqrt((2 * np.pi) ** self._dim)

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
                for jx in range(self._n_clusters):
                    d_column_new[jx] = (
                        np.sum((v_new - self._v[jx, :]) ** 2)
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
            d_column_new = np.zeros(self._n_clusters)
            for jx in range(self._n_clusters):
                # Skip the current i_label index
                if jx == i_label:
                    continue
                d_column_new[jx] = (
                    np.sum((v_new - self._v[jx, :]) ** 2)
                )

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
        Batch parameter update for the (Renyi's) representative Cross Information Potential (rCIP) CVI.
        """

        # Setup the CVI for batch mode
        super()._setup_batch(data)

        # Take the average across all samples, but cast to 1-D vector
        self._mu = np.mean(data, axis=0)
        u = np.unique(labels)
        self._n_clusters = u.size
        # self._n = np.zeros(self._n_clusters, dtype=int)
        self._n = [0 for _ in range(self._n_clusters)]
        self._v = np.zeros((self._n_clusters, self._dim))
        # self._CP = np.zeros(self._n_clusters)
        self._CP = [0 for _ in range(self._n_clusters)]
        self._G = np.zeros((self._n_clusters, self._dim))
        self._D = np.zeros((self._n_clusters, self._n_clusters))
        # self._S = np.zeros(self._n_clusters)
        self._S = [0 for _ in range(self._n_clusters)]

        for ix in range(self._n_clusters):
            # subset_indices = lambda x: labels[x] == ix
            subset_indices = (
                [x for x in range(len(labels)) if labels[x] == ix]
            )
            subset = data[subset_indices, :]
            self._n[ix] = subset.shape[0]
            self._v[ix, :] = np.mean(subset, axis=0)
            diff_x_v = subset - self._v[ix, :] * np.ones((self._n[ix], 1))
            self._CP[ix] = np.sum(diff_x_v ** 2)
            self._S[ix] = self._CP[ix] / self._n[ix]

        for ix in range(self._n_clusters - 1):
            for jx in range(ix + 1, self._n_clusters):
                self._D[ix, jx] = (
                    np.sum((self._v[ix, :] - self._v[jx, :]) ** 2)
                )

        self._D = self._D + np.transpose(self._D)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the (Renyi's) representative Cross Information Potential (rCIP) CVI.
        """

        if self._n_clusters > 1:
            self._R = np.zeros((self._n_clusters, self._n_clusters))
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
            self.criterion_value = 0.0
