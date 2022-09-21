"""
The Centroid-based Silhouette (cSIL) Cluster Validity Index.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# cSIL object definition
class cSIL(_base.CVI):
    """
    The stateful information of the Centroid-based Silhouette (cSIL) Cluster Validity Index.

    References
    ----------
    1. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental Cluster Validity Indices for Hard Partitions: Extensions  and  Comparative Study," ArXiv  e-prints, Feb 2019, arXiv:1902.06711v1 [cs.LG].
    2. P. J. Rousseeuw, "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis," Journal of Computational and Applied Mathematics, vol. 20, pp. 53–65, 1987.
    3. M. Rawashdeh and A. Ralescu, "Center-wise intra-inter silhouettes," in Scalable Uncertainty Management, E. Hüllermeier, S. Link, T. Fober et al., Eds. Berlin, Heidelberg: Springer, 2012, pp. 406–419.
    """

    def __init__(self):
        """
        Centroid-based Silhouette (cSIL) initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # cSIL-specific initialization
        self._S = np.empty([0, 0])   # n_clusters x dim
        self._sil_coefs = []         # dim

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Centroid-based Silhouette (cSIL) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Centroid-based Silhouette (cSIL) CVI.
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
            CP_new = np.inner(sample, sample)
            G_new = sample

            # Compute S_new
            if self._n_clusters == 0:
                S_new = np.zeros([1, 1])
            else:
                S_new = np.zeros((self._n_clusters + 1, self._n_clusters + 1))
                S_new[0:self._n_clusters, 0:self._n_clusters] = self._S
                S_row_new = np.zeros(self._n_clusters + 1)
                S_col_new = np.zeros(self._n_clusters + 1)
                for cl in range(self._n_clusters):
                    # Column "bmu_temp - D_new"
                    C = (
                        CP_new
                        + np.inner(self._v[cl, :], self._v[cl, :])
                        - np.inner(G_new, self._v[cl, :])
                    )
                    S_col_new[cl] = C
                    C = (
                        self._CP[cl]
                        + self._n[cl] * np.inner(v_new, v_new)
                        - 2 * np.inner(self._G[cl, :], v_new)
                    )
                    S_row_new[cl] = C / self._n[cl]
                # Column "ind_minus" - F
                S_col_new[i_label] = 0
                S_row_new[i_label] = S_col_new[i_label]
                S_new[i_label, :] = S_col_new
                S_new[:, i_label] = S_row_new

            # Update 1-D parameters with list appends
            self._n_clusters += 1
            self._n.append(n_new)
            self._CP.append(CP_new)

            # Update 2-D parameters with numpy vstacks
            self._v = np.vstack([self._v, v_new])
            self._G = np.vstack([self._G, G_new])
            self._S = S_new

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self._n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self._v[i_label, :]
                + (1 / n_new) * sample
            )
            # delta_v = self._v[i_label, :] - v_new
            # diff_x_v = sample - v_new
            CP_new = (
                self._CP[i_label]
                + np.inner(sample, sample)
            )
            G_new = (
                self._G[i_label, :]
                + sample
            )
            # Compute S_new
            S_row_new = np.zeros(self._n_clusters)
            S_col_new = np.zeros(self._n_clusters)
            for cl in range(self._n_clusters):
                # Skip the i_label iteration
                if cl == i_label:
                    continue
                # Column "bmu_temp" - D_new
                diff_x_v = sample - self._v[cl, :]
                C = (
                    self._CP[i_label]
                    + np.inner(diff_x_v, diff_x_v)
                    + self._n[i_label] * np.inner(self._v[cl, :], self._v[cl, :])
                    - 2 * np.inner(G_new, self._v[cl, :])
                )
                S_col_new[cl] = C / n_new
                # Row "bmu_temp" - E
                C = (
                    self._CP[cl]
                    + self._n[cl] * np.inner(v_new, v_new)
                    - 2 * np.inner(self._G[cl, :], v_new)
                )
                S_row_new[cl] = C / self._n[cl]

            # Column "ind_minus" - F
            diff_x_v = sample - v_new
            C = (
                self._CP[i_label]
                + np.inner(diff_x_v, diff_x_v)
                + self._n[i_label] * np.inner(v_new, v_new)
                - 2 * np.inner(self._G[i_label, :], v_new)
            )
            S_col_new[i_label] = C / n_new
            S_row_new[i_label] = S_col_new[i_label]

            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._CP[i_label] = CP_new
            self._G[i_label, :] = G_new

            # self._S[:, i_label] = S_col_new
            # self._S[i_label, :] = S_row_new
            self._S[i_label, :] = S_col_new
            self._S[:, i_label] = S_row_new

        # Update the parameters that do not depend on label novelty
        self._n_samples = n_samples_new

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the Centroid-based Silhouette (cSIL) CVI.
        """

        # Setup the CVI for batch mode
        super()._setup_batch(data)

        # Take the average across all samples, but cast to 1-D vector
        u = np.unique(labels)
        self._n_clusters = u.size
        self._n = np.zeros(self._n_clusters, dtype=int)
        self._v = np.zeros((self._n_clusters, self._dim))
        self._CP = np.zeros(self._n_clusters)
        self._S = np.zeros((self._n_clusters, self._n_clusters))
        D = np.zeros((self._n_samples, self._n_samples))
        for ix in range(self._n_clusters):
            subset_indices = (
                [x for x in range(len(labels)) if labels[x] == ix]
            )
            subset = data[subset_indices, :]
            self._n[ix] = subset.shape[0]
            self._v[ix, :] = np.mean(subset, axis=0)

            # Compute CP in case of switching back to incremental mode
            diff_x_v = subset - self._v[ix, :] * np.ones((self._n[ix], 1))
            self._CP[ix] = np.sum(diff_x_v ** 2)

            d_temp = (data - self._v[ix, :] * np.ones((self._n_samples, 1))) ** 2
            D[ix, :] = np.transpose(np.sum(d_temp, axis=1))
            # D[ix, :] = np.sum(d_temp, axis=1)

        for ix in range(self._n_clusters):
            for jx in range(self._n_clusters):
                subset_ind = [x for x in range(len(labels)) if labels[x] == jx]
                self._S[jx, ix] = sum(D[ix, subset_ind]) / self._n[jx]

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Centroid-based Silhouette (cSIL) CVI.
        """

        self._sil_coefs = np.zeros(self._n_clusters)

        if self._n_clusters > 1 and self._S.any():
            for ix in range(self._n_clusters):
                # Same cluster
                a = self._S[ix, ix]
                # Other clusters
                local_S = np.delete(self._S[:, ix], ix)
                b = np.min(local_S)
                self._sil_coefs[ix] = (b - a) / np.maximum(a, b)
            # cSIL index value
            self.criterion_value = np.sum(self._sil_coefs) / self._n_clusters

        else:
            self.criterion_value = 0.0
