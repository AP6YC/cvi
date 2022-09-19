"""
The Davies-Bouldin (DB) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# DB object definition
class DB(_base.CVI):
    """
    The stateful information of the Davies-Bouldin (DB) Cluster Validity Index.

    References
    ----------
    1. D. L. Davies and D. W. Bouldin, "A cluster separation measure," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol. 1, no. 2, pp. 224-227, Feb. 1979.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    def __init__(self):
        """
        Davies-Bouldin (DB) initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # CH-specific initialization
        self.mu = np.zeros([0])     # dim
        self.R = np.zeros([0, 0])   # n_clusters x dim
        self.D = np.zeros([0, 0])   # n_clusters x n_clusters
        self.S = []                 # dim

    @_base.add_docs(_base.setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Davies-Bouldin (DB) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # CH-specific setup
        self.mu = sample

    @_base.add_docs(_base.param_inc_doc)
    def param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Davies-Bouldin (DB) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self.label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self.n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self.n_samples == 0:
            self._setup(sample)
        else:
            self.mu = (1 - 1/n_samples_new) * self.mu + (1/n_samples_new) * sample

        # IF NEW CLUSTER LABEL
        # Correct for python 0-indexing
        if i_label > self.n_clusters - 1:
            n_new = 1
            v_new = sample
            CP_new = 0.0
            G_new = np.zeros(self.dim)
            S_new = 0.0
            if self.n_clusters == 0:
                D_new = np.zeros((1, 1))
            else:
                D_new = np.zeros((self.n_clusters + 1, self.n_clusters + 1))
                D_new[0:self.n_clusters, 0:self.n_clusters] = self.D
                d_column_new = np.zeros(self.n_clusters + 1)
                for jx in range(self.n_clusters):
                    d_column_new[jx] = (
                        np.sum((v_new - self.v[jx, :]) ** 2)
                    )
                D_new[i_label, :] = d_column_new
                # D_new[:, i_label] = np.transpose(d_column_new)
                D_new[:, i_label] = d_column_new

            # Update 1-D parameters with list appends
            self.n_clusters += 1
            self.n.append(n_new)
            self.CP.append(CP_new)
            self.S.append(S_new)

            # Update 2-D parameters with numpy vstacks
            self.v = np.vstack([self.v, v_new])
            self.G = np.vstack([self.G, G_new])
            self.D = D_new

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self.n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self.v[i_label, :]
                + (1 / n_new) * sample
            )
            delta_v = self.v[i_label, :] - v_new
            diff_x_v = sample - v_new
            CP_new = (
                self.CP[i_label]
                + np.inner(diff_x_v, diff_x_v)
                + self.n[i_label] * np.inner(delta_v, delta_v)
                + 2 * np.inner(delta_v, self.G[i_label, :])
            )
            G_new = (
                self.G[i_label, :]
                + diff_x_v
                + self.n[i_label] * delta_v
            )
            S_new = CP_new / n_new
            d_column_new = np.zeros(self.n_clusters)
            for jx in range(self.n_clusters):
                # Skip the current i_label index
                if jx == i_label:
                    continue
                d_column_new[jx] = (
                    np.sum((v_new - self.v[jx, :]) ** 2)
                )

            # Update parameters
            self.n[i_label] = n_new
            self.v[i_label, :] = v_new
            self.CP[i_label] = CP_new
            self.G[i_label, :] = G_new
            self.S[i_label] = S_new
            self.D[i_label, :] = d_column_new
            # self.D[:, i_label] = np.tranpose(d_column_new)
            self.D[:, i_label] = d_column_new

        # Update the parameters that do not depend on label novelty
        self.n_samples = n_samples_new

    @_base.add_docs(_base.param_batch_doc)
    def param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Davies-Bouldin (DB) CVI.
        """

        # Setup the CVI for batch mode
        super()._setup_batch(data)

        # Take the average across all samples, but cast to 1-D vector
        self.mu = np.mean(data, axis=0)
        u = np.unique(labels)
        self.n_clusters = u.size
        # self.n = np.zeros(self.n_clusters, dtype=int)
        self.n = [0 for _ in range(self.n_clusters)]
        self.v = np.zeros((self.n_clusters, self.dim))
        # self.CP = np.zeros(self.n_clusters)
        self.CP = [0 for _ in range(self.n_clusters)]
        self.G = np.zeros((self.n_clusters, self.dim))
        self.D = np.zeros((self.n_clusters, self.n_clusters))
        # self.S = np.zeros(self.n_clusters)
        self.S = [0 for _ in range(self.n_clusters)]

        for ix in range(self.n_clusters):
            # subset_indices = lambda x: labels[x] == ix
            subset_indices = (
                [x for x in range(len(labels)) if labels[x] == ix]
            )
            subset = data[subset_indices, :]
            self.n[ix] = subset.shape[0]
            self.v[ix, :] = np.mean(subset, axis=0)
            diff_x_v = subset - self.v[ix, :] * np.ones((self.n[ix], 1))
            self.CP[ix] = np.sum(diff_x_v ** 2)
            self.S[ix] = self.CP[ix] / self.n[ix]

        for ix in range(self.n_clusters - 1):
            for jx in range(ix + 1, self.n_clusters):
                self.D[ix, jx] = (
                    np.sum((self.v[ix, :] - self.v[jx, :]) ** 2)
                )

        self.D = self.D + np.transpose(self.D)

    @_base.add_docs(_base.evaluate_doc)
    def evaluate(self):
        """
        Criterion value evaluation method for the Davies-Bouldin (DB) CVI.
        """
        self.R = np.zeros((self.n_clusters, self.n_clusters))

        if self.n_clusters > 2:
            for ix in range(self.n_clusters - 1):
                for jx in range(ix + 1, self.n_clusters):
                    self.R[jx, ix] = (
                        (self.S[ix] + self.S[jx]) / self.D[jx, ix]
                    )
            self.R = self.R + np.transpose(self.R)
            self.criterion_value = (
                np.sum(np.max(self.R, axis=0)) / self.n_clusters
            )
        else:
            self.criterion_value = 0.0
