"""
The Calinski-Harabasz (CH) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# CH object definition
class CH(_base.CVI):
    """
    The stateful information of the Calinski-Harabasz (CH) Cluster Validity Index.

    References
    ----------
    1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    2. T. Calinski and J. Harabasz, "A dendrite method for cluster analysis," Communications in Statistics, vol. 3, no. 1, pp. 1-27, 1974.
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    4. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    def __init__(self):
        """
        CH initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # CH-specific initialization
        self.mu = np.zeros([0])     # dim
        self.SEP = np.zeros([0])     # dim
        self.BGSS = 0.0
        self.WGSS = 0.0

        return

    @_base.add_docs(_base.setup_doc)
    def setup(self, sample: np.ndarray) -> None:
        """
        CH setup routine.
        """

        # Run the generic setup routine
        super().setup(sample)

        # CH-specific setup
        self.SEP = np.zeros([self.dim])
        self.mu = sample
        # self.mu = np.zeros([self.dim])

        return

    @_base.add_docs(_base.param_inc_doc)
    def param_inc(self, sample: np.ndarray, label: int) -> None:
        """
        Incremental parameter update for the Calinski-Harabasz (CH) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self.label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self.n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self.n_samples == 0:
            self.setup(sample)
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
            v_new = (1 - 1/n_new) * self.v[i_label, :] + (1/n_new) * sample
            delta_v = self.v[i_label, :] - v_new
            diff_x_v = sample - v_new
            CP_new = (
                self.CP[i_label]
                + np.inner(diff_x_v, diff_x_v)
                + self.n[i_label] * np.inner(delta_v, delta_v)
                + 2*np.inner(delta_v, self.G[i_label, :])
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
        self.SEP = np.array([self.n[ix] * sum((self.v[ix, :] - self.mu)**2) for ix in range(self.n_clusters)])

        return

    @_base.add_docs(_base.param_batch_doc)
    def param_batch(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        Batch parameter update for the Calinski-Harabasz (CH) CVI.
        """

        self.n_samples, self.dim = data.shape
        # Take the average across all samples, but cast to 1-D vector
        self.mu = np.mean(data, axis=0)
        u = np.unique(labels)
        self.n_clusters = u.size
        self.n = np.zeros(self.n_clusters, dtype=int)
        self.v = np.zeros((self.n_clusters, self.dim))
        self.CP = np.zeros(self.n_clusters)
        self.SEP = np.zeros(self.n_clusters)

        for ix in range(self.n_clusters):
            # subset_indices = lambda x: labels[x] == ix
            subset_indices = [x for x in range(len(labels)) if labels[x] == ix]
            subset = data[subset_indices, :]
            self.n[ix] = subset.shape[0]
            self.v[ix, :] = np.mean(subset, axis=0)
            diff_x_v = subset - self.v[ix, :] * np.ones((self.n[ix], 1))
            self.CP[ix] = np.sum(diff_x_v ** 2)
            self.SEP[ix] = self.n[ix] * np.sum((self.v[ix, :] - self.mu) ** 2)

        return

    @_base.add_docs(_base.evaluate_doc)
    def evaluate(self) -> None:
        """
        Criterion value evaluation method for the Calinski-Harabasz (CH) CVI.
        """

        if self.n_clusters > 2:
            # Within group sum of scatters
            self.WGSS = sum(self.CP)
            # Between groups sum of scatters
            self.BGSS = sum(self.SEP)
            # CH index value
            self.criterion_value = (
                (self.BGSS / self.WGSS)
                * ((self.n_samples - self.n_clusters) / (self.n_clusters - 1))
            )
        else:
            self.BGSS = 0.0
            self.criterion_value = 0.0

        return
