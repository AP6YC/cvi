"""
Generalized Dunn's Index 53 (GD53) Cluster Validity Index.

References
----------
1. A. Ibrahim, J. M. Keller, and J. C. Bezdek, "Evaluating Evolving Structure in Streaming Data With Modified Dunn's Indices," IEEE Transactions on Emerging Topics in Computational Intelligence, pp. 1-12, 2019.
2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML].
3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
4. J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters," J. Cybern., vol. 3, no. 3 , pp. 32-57, 1973.
5. J. C. Bezdek and N. R. Pal, "Some new indexes of cluster validity," IEEE Trans. Syst., Man, and Cybern., vol. 28, no. 3, pp. 301-315, Jun. 1998.
"""

# Custom imports
import numpy as np

# Local imports
from . import _base


# GD53 object definition
class GD53(_base.CVI):
    """
    Generalized Dunn's Index 53 (GD53) Cluster Validity Index.

    References
    ----------
    1. A. Ibrahim, J. M. Keller, and J. C. Bezdek, "Evaluating Evolving Structure in Streaming Data With Modified Dunn's Indices," IEEE Transactions on Emerging Topics in Computational Intelligence, pp. 1-12, 2019.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    4. J. C. Dunn, "A fuzzy relative of the ISODATA process and its use in detecting compact well-separated clusters," J. Cybern., vol. 3, no. 3 , pp. 32-57, 1973.
    5. J. C. Bezdek and N. R. Pal, "Some new indexes of cluster validity," IEEE Trans. Syst., Man, and Cybern., vol. 28, no. 3, pp. 301-315, Jun. 1998.
    """

    info = _base.CVIInfo(
        name="Generalized Dunn's 53",
        name_short="GD53",
        index_min=0.0,
        index_max=np.inf,
        optimality="max",
        batch=True,
        incremental=True,
        merge=True,
        remove=True,
        split=True,
        backends=("numpy", "numba"),
    )
    _supports_numba = True
    _supports_remove_merge = True
    _uses_compactness_stats = True

    def __init__(self, *, backend="numpy"):
        """
        Generalized Dunn's Index 53 (GD53) initialization routine.

        Parameters
        ----------
        backend : {"numpy", "numba"}, default="numpy"
            Select the numerical backend. Numba is loaded on demand.
        """

        # Run the base initialization
        super().__init__(backend=backend)

        # GD53-specific initialization
        self._mu = np.zeros([0])     # dim
        self._D = np.zeros([0, 0])   # n_clusters x n_clusters
        self._inter = 0.0
        self._intra = 0.0

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        Generalized Dunn's Index 53 (GD53) setup routine.
        """

        # Run the generic setup routine
        super()._setup(sample)

        # GD53-specific setup
        self._mu = sample

    @staticmethod
    def _dispersion_row(compactness, counts, compactness_new, count_new):
        """Return GD53 dispersion from one cluster to every cluster."""
        return ((compactness_new + np.asarray(compactness))
                / (count_new + np.asarray(counts)))

    @classmethod
    def _dispersion_matrix(cls, compactness, counts):
        """Return the symmetric GD53 dispersion matrix with a zero diagonal."""
        compactness = np.asarray(compactness)
        counts = np.asarray(counts)
        matrix = ((compactness[:, None] + compactness[None, :])
                  / (counts[:, None] + counts[None, :]))
        np.fill_diagonal(matrix, 0.0)
        return matrix

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the Generalized Dunn's Index 53 (GD53) CVI.
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
                if self._n_clusters < 64:
                    for jx in range(self._n_clusters):
                        d_column_new[jx] = (
                            (CP_new + self._CP[jx]) / (n_new + self._n[jx])
                        )
                else:
                    d_column_new[:-1] = self._dispersion_row(
                        self._CP, self._n, CP_new, n_new,
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
            if self._n_clusters < 64:
                d_column_new = np.zeros(self._n_clusters)
                for jx in range(self._n_clusters):
                    if jx != i_label:
                        d_column_new[jx] = (
                            (CP_new + self._CP[jx]) / (n_new + self._n[jx])
                        )
            else:
                d_column_new = self._dispersion_row(
                    self._CP, self._n, CP_new, n_new,
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
        Batch parameter update for the Generalized Dunn's Index 53 (GD53) CVI.
        """

        self._setup_batch_statistics(data, labels)
        self._mu = np.mean(data, axis=0)
        self._D = self._dispersion_matrix(self._CP, self._n)

    def _rebuild_after_operation(self):
        """Rebuild pairwise dispersion after a structural operation."""

        if self._n_clusters == 0:
            self._mu = np.zeros(0)
            self._D = np.zeros((0, 0))
            self._inter = 0.0
            self._intra = 0.0
            return

        self._D = self._dispersion_matrix(self._CP, self._n)
        if self._n_clusters < 2:
            self._inter = 0.0
            self._intra = 0.0

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the Generalized Dunn's Index 53 (GD53) CVI.
        """

        if self._n_clusters > 1:
            self._intra = 2 * np.max(np.divide(self._CP, self._n))
            # Between-group measure of separation/isolation
            self._inter = self._backend.minimum_off_diagonal(self._D)
            # GD53 index value
            if self._intra > 0.0:
                self.criterion_value = self._inter / self._intra
            else:
                self.criterion_value = np.nan
        else:
            self.criterion_value = np.nan
