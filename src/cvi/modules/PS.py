"""
The Partition Separation (PS) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# PS object definition
class PS(_base.CVI):
    """
    The Partition Separation (PS) Cluster Validity Index.

    References
    ----------
    1. Miin-Shen Yang and Kuo-Lung Wu, "A new validity index for fuzzy clustering," 10th IEEE International Conference on Fuzzy Systems. (Cat. No.01CH37297), Melbourne, Victoria, Australia, 2001, pp. 89-92, vol.1.
    2. E. Lughofer, "Extensions of vector quantization for incremental clustering," Pattern Recognit., vol. 41, no. 3, pp. 995-1011, 2008.
    """

    def __init__(self):
        """
        Partition Separation (PS) initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # PS-specific initialization
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._v_bar = []
        self._beta_t = 0.0
        self._PS_i = np.zeros(0)

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Partition Separation (PS) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # CH-specific setup
        # Delete unused members
        del self._G
        del self._CP

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Partition Separation (PS) CVI.
        """

        # Get the internal label corresponding to the provided label
        i_label = self._label_map.get_internal_label(label)

        # Increment the local number of samples count
        n_samples_new = self._n_samples + 1

        # Check if the module has been setup, then set the mu accordingly
        if self._n_samples == 0:
            self._setup(sample)

        # IF NEW CLUSTER LABEL
        # Correct for python 0-indexing
        if i_label > self._n_clusters - 1:
            n_new = 1
            v_new = sample
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

            # Update 2-D parameters with numpy vstacks
            self._v = np.vstack([self._v, v_new])
            self._D = D_new

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self._n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self._v[i_label, :]
                + (1 / n_new) * sample
            )
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
            self._D[i_label, :] = d_column_new
            self._D[:, i_label] = d_column_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Partition Separation (PS) CVI.
        """

        # Setup the CVI for batch mode
        super()._setup_batch(data)

        # Take the average across all samples, but cast to 1-D vector
        self._mu = np.mean(data, axis=0)
        u = np.unique(labels)
        self._n_clusters = u.size
        self._n = [0 for _ in range(self._n_clusters)]
        self._v = np.zeros((self._n_clusters, self._dim))
        self._D = np.zeros((self._n_clusters, self._n_clusters))

        for ix in range(self._n_clusters):
            subset_indices = (
                [x for x in range(len(labels)) if labels[x] == ix]
            )
            subset = data[subset_indices, :]
            self._n[ix] = subset.shape[0]
            self._v[ix, :] = np.mean(subset, axis=0)
            # diff_x_v = subset - self._v[ix, :] * np.ones((self._n[ix], 1))

        for ix in range(self._n_clusters - 1):
            for jx in range(ix + 1, self._n_clusters):
                self._D[ix, jx] = (
                    np.sum((self._v[ix, :] - self._v[jx, :]) ** 2)
                )

        self._D = self._D + np.transpose(self._D)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Partition Separation (PS) CVI.
        """

        if self._n_clusters > 1:
            self._v_bar = np.mean(self._v, axis=0)
            self._beta_t = 0.0
            self._PS_i = np.zeros(self._n_clusters)
            for ix in range(self._n_clusters):
                delta_v = self._v[ix, :] - self._v_bar
                self._beta_t = self._beta_t + np.inner(delta_v, delta_v)
            self._beta_t /= self._n_clusters
            n_max = max(self._n)
            for ix in range(self._n_clusters):
                d = self._D[ix, :]
                d = np.delete(d, ix)
                self._PS_i[ix] = (
                    (self._n[ix] / n_max)
                    - np.exp(-np.min(d) / self._beta_t)
                )
            self.criterion_value = np.sum(self._PS_i)
        else:
            self.criterion_value = 0.0
