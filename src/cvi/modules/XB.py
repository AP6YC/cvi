"""
The Xie-Beni (XB) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# XB object definition
class XB(_base.CVI):
    """
    The stateful information of the Xie-Beni (XB) Cluster Validity Index.

    References
    ----------
    1. X. L. Xie and G. Beni, "A Validity Measure for Fuzzy Clustering," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 8, pp. 841-847, 1991.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    def __init__(self):
        """
        XB initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # XB-specific initialization
        self.mu = np.zeros([0])     # dim
        self.SEP = np.zeros([0])     # dim
        self.BGSS = 0.0
        self.WGSS = 0.0

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Xie-Beni (XB) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # XB-specific setup
        self.SEP = np.zeros([self.dim])
        self.mu = sample

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Xie-Beni (XB) CVI.
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

            # Update 1-D parameters with list appends
            self.n_clusters += 1
            self.n.append(n_new)
            self.CP.append(CP_new)

            # Update 2-D parameters with numpy vstacks
            self.v = np.vstack([self.v, v_new])
            self.G = np.vstack([self.G, G_new])

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
            # Update parameters
            self.n[i_label] = n_new
            self.v[i_label, :] = v_new
            self.CP[i_label] = CP_new
            self.G[i_label, :] = G_new

        # Update the parameters that do not depend on label novelty
        self.n_samples = n_samples_new
        # self.mu = mu_new
        self.SEP = np.array([
            self.n[ix] * sum((self.v[ix, :] - self.mu)**2)
            for ix in range(self.n_clusters)
        ])

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Xie-Beni (XB) CVI.
        """

        # Setup the CVI for batch mode
        super()._setup_batch(data)

        # Take the average across all samples, but cast to 1-D vector
        self.mu = np.mean(data, axis=0)
        u = np.unique(labels)
        self.n_clusters = u.size
        self.n = np.zeros(self.n_clusters, dtype=int)
        self.v = np.zeros((self.n_clusters, self.dim))
        self.CP = np.zeros(self.n_clusters)
        self.SEP = np.zeros(self.n_clusters)

        for ix in range(self.n_clusters):
            subset_indices = (
                [x for x in range(len(labels)) if labels[x] == ix]
            )
            subset = data[subset_indices, :]
            self.n[ix] = subset.shape[0]
            self.v[ix, :] = np.mean(subset, axis=0)
            diff_x_v = subset - self.v[ix, :] * np.ones((self.n[ix], 1))
            self.CP[ix] = np.sum(diff_x_v ** 2)
            self.SEP[ix] = self.n[ix] * np.sum((self.v[ix, :] - self.mu) ** 2)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Xie-Beni (XB) CVI.
        """

        if self.n_clusters > 2:
            # Within group sum of scatters
            self.WGSS = sum(self.CP)
            # Between groups sum of scatters
            self.BGSS = sum(self.SEP)
            # XB index value
            self.criterion_value = (
                (self.WGSS / self.BGSS) * self.n_clusters
            )
        else:
            self.BGSS = 0.0
            self.criterion_value = 0.0
