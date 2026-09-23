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
from . import _base


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
        batch=True,
        incremental=True,
        merge=True,
        remove=True,
        split=True,
        backends=("numpy", "numba", "jax"),
    )
    _supports_numba = True
    _supports_jax = True
    _supports_remove_merge = True
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

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Xie-Beni (XB) CVI.
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
            d_column_new = self._backend.centroid_distances(
                self._v, v_new,
            )
            d_column_new[i_label] = 0.0

            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._CP[i_label] = CP_new
            self._G[i_label, :] = G_new
            self._D[i_label, :] = d_column_new
            self._D[:, i_label] = d_column_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new

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
            # Assume a symmetric dimension
            dim = self._D.shape[0]
            # self.values = (
            #     [self._D[i, j] for i in range(dim) for j in range(dim) if j > i]
            # )
            values = self._D[np.triu_indices(dim, k=1)]
            self._SEP = np.min(values)
            if self._SEP > 0.0:
                # XB index value
                self.criterion_value = (
                    self._WGSS / (self._n_samples * self._SEP)
                )
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
