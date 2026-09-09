"""
(Renyi's) representative Cross Information Potential (rCIP) Cluster Validity Index.

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


# rCIP object definition
class rCIP(_base.CVI):
    """
    (Renyi's) representative Cross Information Potential (rCIP) Cluster Validity Index.

    References
    ----------
    1. D. L. Davies and D. W. Bouldin, "A cluster separation measure," IEEE Transaction on Pattern Analysis and Machine Intelligence, vol. 1, no. 2, pp. 224-227, Feb. 1979.
    2. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, and J. Bailey, "Online Cluster Validity Indices for Streaming Data," ArXiv e-prints, 2018, arXiv:1801.02937v1 [stat.ML]. [Online].
    3. M. Moshtaghi, J. C. Bezdek, S. M. Erfani, C. Leckie, J. Bailey, "Online cluster validity indices for performance monitoring of streaming data clustering," Int. J. Intell. Syst., pp. 1-23, 2018.
    """

    info = _base.CVIInfo(
        name="Representative Cross Information Potential",
        name_short="rCIP",
        index_min=0.0,
        index_max=np.inf,
        optimality="min"
    )
    _supports_remove_merge = True

    def __init__(self):
        """
        (Renyi's) representative Cross Information Potential (rCIP) initialization routine.
        """

        # Run the base initialization
        super().__init__()

        # rCIP-specific initialization
        self._D = np.zeros([0, 0])              # n_clusters x n_clusters
        self._sigma = np.zeros([0, 0, 0])       # dim x dim x n_clusters
        self._delta_term = np.zeros([0, 0])     # dim x dim
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
            sigma_new = self._delta_term
            if self._n_clusters == 0:
                D_new = np.zeros((1, 1))
            else:
                D_new = np.zeros((self._n_clusters + 1, self._n_clusters + 1))
                D_new[0:self._n_clusters, 0:self._n_clusters] = self._D
                d_column_new = np.zeros(self._n_clusters + 1)
                for jx in range(self._n_clusters):
                    diff_m = v_new - self._v[jx, :]
                    sigma_q = sigma_new + self._sigma[:, :, jx]
                    d_column_new[jx] = (
                        self._constant
                        * (1 / np.sqrt(np.linalg.det(sigma_q)))
                        * np.exp(
                            -0.5 * diff_m @ np.linalg.inv(sigma_q) @ diff_m
                        )
                    )
                D_new[i_label, :] = d_column_new
                D_new[:, i_label] = d_column_new

            # Update 1-D parameters with list appends
            self._n_clusters += 1
            self._n.append(n_new)

            # Update 2-D parameters with numpy vstack
            self._v = np.vstack([self._v, v_new])
            self._D = D_new

            # Update 3-D parameters with numpy dstack
            self._sigma = np.dstack((self._sigma, sigma_new))

        # ELSE OLD CLUSTER LABEL
        else:
            n_new = self._n[i_label] + 1
            v_new = (
                (1 - 1 / n_new) * self._v[i_label, :]
                + (1 / n_new) * sample
            )
            diff_x_v = sample - self._v[i_label, :]
            sigma_new = (
                ((n_new - 2) / (n_new - 1))
                * (self._sigma[:, :, i_label] - self._delta_term)
                + (1 / n_new) * (np.outer(diff_x_v, diff_x_v))
                + self._delta_term
            )
            d_column_new = np.zeros(self._n_clusters)
            for jx in range(self._n_clusters):
                # Skip the current i_label index
                if jx == i_label:
                    continue
                diff_m = v_new - self._v[jx, :]
                sigma_q = sigma_new + self._sigma[:, :, jx]
                d_column_new[jx] = (
                    self._constant
                    * (1 / np.sqrt(np.linalg.det(sigma_q)))
                    * np.exp(
                        -0.5 * diff_m @ np.linalg.inv(sigma_q) @ diff_m
                    )
                )

            # Update parameters
            self._n[i_label] = n_new
            self._v[i_label, :] = v_new
            self._sigma[:, :, i_label] = sigma_new
            self._D[i_label, :] = d_column_new
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
        # TODO include this in a local setup batch method
        epsilon = 12.0
        delta = 10.0 ** (-epsilon / self._dim)
        self._delta_term = np.eye(self._dim) * delta
        self._constant = 1 / np.sqrt((2 * np.pi) ** self._dim)

        # Take the average across all samples, but cast to 1-D vector
        u = self._setup_batch_labels(labels)
        self._n_clusters = len(u)
        self._n = [0 for _ in range(self._n_clusters)]
        self._v = np.zeros((self._n_clusters, self._dim))
        self._G = np.zeros((0, self._dim))
        self._sigma = np.zeros((self._dim, self._dim, self._n_clusters))
        self._D = np.zeros((self._n_clusters, self._n_clusters))

        for ix, external_label in enumerate(u):
            subset_indices = (
                [x for x in range(len(labels))
                 if labels[x] == external_label]
            )
            subset = data[subset_indices, :]
            self._n[ix] = subset.shape[0]
            self._v[ix, :] = np.mean(subset, axis=0)
            if self._n[ix] > 1:
                self._sigma[:, :, ix] = (
                    (1 / (self._n[ix] - 1)) * (
                        np.transpose(subset) @ subset
                        - self._n[ix] * np.outer(self._v[ix, :], self._v[ix, :])
                    ) + self._delta_term
                )
            else:
                self._sigma[:, :, ix] = self._delta_term

        for ix in range(self._n_clusters - 1):
            for jx in range(ix + 1, self._n_clusters):
                diff_m = self._v[ix, :] - self._v[jx, :]
                sigma_q = self._sigma[:, :, ix] + self._sigma[:, :, jx]
                self._D[ix, jx] = (
                    self._constant
                    * (1 / np.sqrt(np.linalg.det(sigma_q)))
                    * np.exp(
                        -0.5 * diff_m @ np.linalg.inv(sigma_q) @ diff_m
                    )
                )

        self._D = self._D + np.transpose(self._D)

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Symmetrize covariance and clip insignificant negative eigenvalues."""

        covariance = (covariance + covariance.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 1e-10 * max(1.0, np.linalg.norm(covariance, ord=2))

        if np.min(eigenvalues) < -tolerance:
            raise ValueError(
                "The requested operation produces invalid covariance; "
                "check the supplied sample and cluster label"
            )

        eigenvalues = np.maximum(eigenvalues, 0.0)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _delete_cluster(self, label: int, i_label: int):
        """Delete one rCIP cluster and compact its internal label."""

        self._sigma = np.delete(self._sigma, i_label, axis=2)
        super()._delete_cluster(label, i_label)

    def _remove(self, sample: np.ndarray, label: int, i_label: int):
        """Remove a sample from rCIP's mean and covariance statistics."""

        n_old = self._n[i_label]
        v_old = self._v[i_label, :].copy()
        n_samples_new = self._n_samples - 1

        if n_old == 1:
            self._validate_singleton_removal(sample, v_old)

            self._delete_cluster(label, i_label)
            self._n_samples = n_samples_new

            if n_samples_new == 0:
                self._clear_common_state()
                self._CP = None
                self._D = np.zeros((0, 0))
                self._sigma = np.zeros((0, 0, 0))
                self._delta_term = np.zeros((0, 0))
                self._constant = 0.0

            self._rebuild_after_operation()
            return

        n_new = n_old - 1
        v_new = (n_old * v_old - sample) / n_new

        if n_new == 1:
            sigma_new = self._delta_term.copy()
        else:
            covariance_old = (
                self._sigma[:, :, i_label] - self._delta_term
            )
            difference = sample - v_old
            covariance_new = (
                ((n_old - 1) / (n_old - 2)) * covariance_old
                - (n_old / ((n_old - 1) * (n_old - 2)))
                * np.outer(difference, difference)
            )
            covariance_new = self._stabilize_covariance(covariance_new)
            sigma_new = covariance_new + self._delta_term

        self._n[i_label] = n_new
        self._v[i_label, :] = v_new
        self._sigma[:, :, i_label] = sigma_new
        self._n_samples = n_samples_new
        self._rebuild_after_operation()

    def _merge(
        self,
        target_label: int,
        source_label: int,
        target_i: int,
        source_i: int,
    ):
        """Merge two rCIP mean and covariance summaries."""

        n_target = self._n[target_i]
        n_source = self._n[source_i]
        n_new = n_target + n_source
        v_target = self._v[target_i, :].copy()
        v_source = self._v[source_i, :].copy()
        v_new = (n_target * v_target + n_source * v_source) / n_new
        covariance_target = self._sigma[:, :, target_i] - self._delta_term
        covariance_source = self._sigma[:, :, source_i] - self._delta_term
        difference = v_source - v_target
        covariance_new = (
            ((n_target - 1) / (n_new - 1)) * covariance_target
            + ((n_source - 1) / (n_new - 1)) * covariance_source
            + (n_target * n_source / (n_new * (n_new - 1)))
            * np.outer(difference, difference)
        )
        covariance_new = self._stabilize_covariance(covariance_new)
        sigma_new = covariance_new + self._delta_term

        self._n[target_i] = n_new
        self._v[target_i, :] = v_new
        self._sigma[:, :, target_i] = sigma_new
        self._delete_cluster(source_label, source_i)
        self._rebuild_after_operation()

    def _rebuild_after_operation(self):
        """Rebuild pairwise representative information potentials."""

        if self._n_clusters == 0:
            self._D = np.zeros((0, 0))
            return

        def information_potential(ix, jx):
            difference = self._v[ix, :] - self._v[jx, :]
            sigma_q = self._sigma[:, :, ix] + self._sigma[:, :, jx]
            return (
                self._constant
                * (1 / np.sqrt(np.linalg.det(sigma_q)))
                * np.exp(
                    -0.5
                    * difference
                    @ np.linalg.inv(sigma_q)
                    @ difference
                )
            )

        self._D = self._pairwise_matrix(
            self._n_clusters,
            information_potential,
        )

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for the (Renyi's) representative Cross Information Potential (rCIP) CVI.
        """

        dim = self._D.shape[0]
        if dim > 1:
            values = self._D[np.triu_indices(dim, k=1)]
            self.criterion_value = np.sum(values)

        else:
            self.criterion_value = 0.0
