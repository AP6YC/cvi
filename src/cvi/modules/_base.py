"""
Utilities that are common across all CVI objects.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

# Standard library imports
from typing import (
    Callable,
    Optional,
    Union
)
from abc import abstractmethod

from dataclasses import dataclass
from typing import ClassVar

# Custom imports
import numpy as np

# --------------------------------------------------------------------------- #
# CLASSES
# --------------------------------------------------------------------------- #

@dataclass
class CVIInfo:
    name: str
    name_short: str
    index_min: float
    index_max: float
    optimality: str

class LabelMap():
    """
    Internal map between labels and the incremental CVI categories.
    """

    def __init__(self):
        self.map = dict()

    def get_internal_label(self, label: int) -> int:
        """
        Gets the internal label and updates the label map if the label is new.
        """

        # Initialize the internal label
        internal_label = None

        # If the label is in the map, return that
        if label in self.map:
            internal_label = self.map[label]
        # Otherwise, create an incremented new label and return that
        else:
            # Correct for python zero-indexing by not including the +1
            internal_label = len(self.map.items())
            self.map[label] = internal_label

        return internal_label

    def get_existing_label(self, label: int) -> int:
        """
        Gets an existing internal label without modifying the label map.

        Raises
        ------
        ValueError
            If the external label is not present in the map.
        """

        if label not in self.map:
            raise ValueError(f"Unknown cluster label: {label}")

        return self.map[label]

    def remove_label(self, label: int) -> int:
        """
        Removes a label and compacts all following internal labels.

        Returns
        -------
        int
            The removed internal label.
        """

        internal_label = self.get_existing_label(label)
        del self.map[label]

        for external_label, mapped_label in self.map.items():
            if mapped_label > internal_label:
                self.map[external_label] = mapped_label - 1

        return internal_label


class CVI():
    """
    Superclass containing elements shared between all CVIs.
    """

    info: ClassVar[CVIInfo]
    _supports_remove_merge: ClassVar[bool] = False
    _uses_compactness_stats: ClassVar[bool] = False

    def __init__(self):
        """
        CVI base class initialization method.
        """

        self._label_map = LabelMap()
        self._dim = 0
        self._n_samples = 0
        self._n = []                 # dim
        self._v = np.zeros([0, 0])   # n_clusters x dim
        self._CP = []                # dim
        self._G = np.zeros([0, 0])   # n_clusters x dim
        self._n_clusters = 0
        self.criterion_value = 0.0
        self._is_defined = False
        self._is_setup = False

    @property
    def is_defined(self) -> bool:
        """Whether the latest criterion value is defined.

        Undefined states return ``0.0`` as a sentinel. A criterion value of
        ``0.0`` can still be defined, so callers should inspect this property
        when consuming values during initialization or after structural
        updates.
        """

        return self._is_defined

    def _setup(self, sample: np.ndarray):
        """
        Common CVI procedure for incremental setup.

        Parameters
        ----------
        data : numpy.ndarray
            Sample vector of features.
        """

        # Infer the dimension as the length of the provided sample
        self._dim = len(sample)

        # Set the sizes of common arrays for consistent appending
        self._v = np.zeros([0, self._dim])
        self._G = np.zeros([0, self._dim])

        # Declare that the CVI is internally setup
        self._is_setup = True

    def _setup_batch(self, data: np.ndarray):
        """
        Common CVI procedure for batch setup.

        Parameters
        ----------
        data : np.ndarray
            A batch of samples with some feature dimension.
        """

        # Infer the data dimension and number of samples
        self._n_samples, self._dim = data.shape
        self._is_setup = True

    def _setup_batch_labels(self, labels: np.ndarray):
        """Populate the label map and return labels in first-seen order."""

        self._label_map = LabelMap()
        unique_labels = []
        for label in np.asarray(labels):
            external_label = label.item() if hasattr(label, "item") else label
            if external_label not in self._label_map.map:
                self._label_map.get_internal_label(external_label)
                unique_labels.append(external_label)

        return unique_labels

    @abstractmethod
    def _param_inc(self, sample: np.ndarray, label: int):
        raise NotImplementedError

    @abstractmethod
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self):
        raise NotImplementedError

    def _require_operations(self):
        """Validate that structural operations are supported and available."""

        if not self._supports_remove_merge:
            raise NotImplementedError(
                f"{type(self).__name__} does not support remove, merge, or split"
            )

        if not self._is_setup:
            raise ValueError(
                "Remove, merge, and split require an initialized CVI"
            )

    def _validate_sample(self, sample: np.ndarray) -> np.ndarray:
        """Validate and normalize a sample used by a structural operation."""

        sample = np.asarray(sample, dtype=float)

        if sample.ndim != 1:
            raise ValueError("Remove requires a one-dimensional sample")

        if sample.shape[0] != self._dim:
            raise ValueError(
                f"Expected a sample with {self._dim} features, "
                f"received {sample.shape[0]}"
            )

        if not np.all(np.isfinite(sample)):
            raise ValueError("Remove requires a sample containing finite values")

        return sample

    @staticmethod
    def _nonnegative_or_error(value: float, scale: float, name: str) -> float:
        """Clip floating-point noise or reject a materially negative statistic."""

        tolerance = 1e-10 * max(1.0, abs(scale))
        if value < -tolerance:
            raise ValueError(
                f"The requested operation produces invalid {name}; "
                "check the supplied sample and cluster label"
            )

        return max(0.0, float(value))

    @staticmethod
    def _validate_singleton_removal(
        sample: np.ndarray,
        centroid: np.ndarray,
    ):
        """Validate that a removed sample matches a singleton centroid."""

        if not np.allclose(sample, centroid, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "The supplied sample does not match the singleton cluster"
            )

    def _validate_split_inputs(
        self,
        retained_label: int,
        new_label: int,
        count: int,
        centroid: np.ndarray,
        compactness: Optional[float],
        covariance: Optional[np.ndarray],
    ):
        """Validate and normalize the sufficient statistics for a split."""

        retained_i = self._label_map.get_existing_label(retained_label)

        if new_label in self._label_map.map:
            raise ValueError(
                f"Split requires an unused new cluster label: {new_label}"
            )

        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count,
            (int, np.integer),
        ):
            raise ValueError("Split count must be a positive integer")

        count = int(count)
        if count < 1:
            raise ValueError("Split count must be a positive integer")

        if count >= self._n[retained_i]:
            raise ValueError(
                "Split count must be smaller than the retained cluster count"
            )

        centroid = np.asarray(centroid, dtype=float)
        if centroid.ndim != 1:
            raise ValueError("Split centroid must be one-dimensional")

        if centroid.shape[0] != self._dim:
            raise ValueError(
                f"Expected a centroid with {self._dim} features, "
                f"received {centroid.shape[0]}"
            )

        if not np.all(np.isfinite(centroid)):
            raise ValueError("Split centroid must contain finite values")

        if compactness is not None:
            compactness_array = np.asarray(compactness, dtype=float)
            if compactness_array.ndim != 0:
                raise ValueError("Split compactness must be a scalar")

            compactness = float(compactness_array)
            if not np.isfinite(compactness):
                raise ValueError("Split compactness must be finite")

            compactness = self._nonnegative_or_error(
                compactness,
                compactness,
                "split compactness",
            )

        if covariance is not None:
            covariance = np.asarray(covariance, dtype=float)
            expected_shape = (self._dim, self._dim)
            if covariance.shape != expected_shape:
                raise ValueError(
                    f"Expected covariance with shape {expected_shape}, "
                    f"received {covariance.shape}"
                )

            if not np.all(np.isfinite(covariance)):
                raise ValueError("Split covariance must contain finite values")

            if not np.allclose(
                covariance,
                covariance.T,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError("Split covariance must be symmetric")

            covariance = (covariance + covariance.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            tolerance = 1e-10 * max(
                1.0,
                np.linalg.norm(covariance, ord=2),
            )
            if np.min(eigenvalues) < -tolerance:
                raise ValueError(
                    "Split covariance must be positive semidefinite"
                )

            eigenvalues = np.maximum(eigenvalues, 0.0)
            covariance = (eigenvectors * eigenvalues) @ eigenvectors.T

        if count == 1:
            if compactness is not None and compactness > 1e-10:
                raise ValueError("Singleton split compactness must be zero")
            compactness = 0.0

            if (
                covariance is not None
                and np.linalg.norm(covariance, ord=2) > 1e-10
            ):
                raise ValueError("Singleton split covariance must be zero")
            covariance = np.zeros((self._dim, self._dim))

        return retained_i, count, centroid, compactness, covariance

    @staticmethod
    def _delete_vector_entry(values, index: int):
        """Delete one entry from either a list or a NumPy vector."""

        if isinstance(values, list):
            del values[index]
            return values

        return np.delete(values, index)

    @staticmethod
    def _pairwise_matrix(n_clusters: int, measure: Callable) -> np.ndarray:
        """Build a symmetric pairwise cluster matrix."""

        matrix = np.zeros((n_clusters, n_clusters))
        for ix in range(n_clusters - 1):
            for jx in range(ix + 1, n_clusters):
                value = measure(ix, jx)
                matrix[ix, jx] = value
                matrix[jx, ix] = value

        return matrix

    def _delete_cluster(self, label: int, i_label: int):
        """Delete universally shared state for one cluster."""

        self._n = self._delete_vector_entry(self._n, i_label)
        self._v = np.delete(self._v, i_label, axis=0)
        self._label_map.remove_label(label)
        self._n_clusters -= 1

    def _delete_common_cluster(self, label: int, i_label: int):
        """Delete a compactness-based cluster and compact its internal label."""

        self._CP = self._delete_vector_entry(self._CP, i_label)
        self._G = np.delete(self._G, i_label, axis=0)
        self._delete_cluster(label, i_label)

    def _clear_common_state(self):
        """Return the common CVI state to its pre-initialization values."""

        self._label_map = LabelMap()
        self._dim = 0
        self._n_samples = 0
        self._n = []
        self._v = np.zeros([0, 0])
        self._CP = []
        self._G = np.zeros([0, 0])
        self._n_clusters = 0
        self.criterion_value = 0.0
        self._is_defined = False
        self._is_setup = False

    def _rebuild_after_operation(self):
        """Rebuild CVI-specific derived state after a structural operation."""

        raise NotImplementedError

    def _remove(self, sample: np.ndarray, label: int, i_label: int):
        """Remove a sample from a compactness-based CVI."""

        if not self._uses_compactness_stats:
            raise NotImplementedError

        n_old = self._n[i_label]
        v_old = self._v[i_label, :].copy()
        n_samples_new = self._n_samples - 1

        if n_old == 1:
            self._validate_singleton_removal(sample, v_old)

            mu_new = None
            if n_samples_new > 0:
                mu_new = (
                    self._n_samples * self._mu - sample
                ) / n_samples_new

            self._delete_common_cluster(label, i_label)
            self._n_samples = n_samples_new

            if n_samples_new == 0:
                self._clear_common_state()
            else:
                self._mu = mu_new

            self._rebuild_after_operation()
            return

        n_new = n_old - 1
        v_new = (n_old * v_old - sample) / n_new
        distance = float(np.inner(sample - v_old, sample - v_old))
        correction = (n_old / n_new) * distance
        CP_new = self._nonnegative_or_error(
            self._CP[i_label] - correction,
            max(abs(self._CP[i_label]), correction),
            "cluster compactness",
        )
        mu_new = (
            self._n_samples * self._mu - sample
        ) / n_samples_new

        self._n[i_label] = n_new
        self._v[i_label, :] = v_new
        self._CP[i_label] = CP_new
        self._G[i_label, :] = np.zeros(self._dim)
        self._n_samples = n_samples_new
        self._mu = mu_new
        self._rebuild_after_operation()

    def _merge(
        self,
        target_label: int,
        source_label: int,
        target_i: int,
        source_i: int,
    ):
        """Merge two clusters in a compactness-based CVI."""

        if not self._uses_compactness_stats:
            raise NotImplementedError

        n_target = self._n[target_i]
        n_source = self._n[source_i]
        n_new = n_target + n_source
        v_target = self._v[target_i, :].copy()
        v_source = self._v[source_i, :].copy()
        v_new = (n_target * v_target + n_source * v_source) / n_new
        difference = v_source - v_target
        CP_new = (
            self._CP[target_i]
            + self._CP[source_i]
            + (n_target * n_source / n_new)
            * np.inner(difference, difference)
        )

        if not np.isfinite(CP_new):
            raise ValueError("The requested merge produces invalid compactness")

        self._n[target_i] = n_new
        self._v[target_i, :] = v_new
        self._CP[target_i] = max(0.0, float(CP_new))
        self._G[target_i, :] = np.zeros(self._dim)
        self._delete_common_cluster(source_label, source_i)
        self._rebuild_after_operation()

    def _split(
        self,
        new_label: int,
        retained_i: int,
        count: int,
        centroid: np.ndarray,
        compactness: Optional[float],
        covariance: Optional[np.ndarray],
    ):
        """Split sufficient statistics from a compactness-based CVI."""

        if not self._uses_compactness_stats:
            raise NotImplementedError

        if compactness is None:
            raise ValueError(
                f"{type(self).__name__} split requires compactness"
            )

        n_parent = self._n[retained_i]
        n_remainder = n_parent - count
        v_parent = self._v[retained_i, :].copy()
        v_remainder = (
            n_parent * v_parent - count * centroid
        ) / n_remainder
        difference = centroid - v_parent
        correction = (
            n_parent * count / n_remainder
        ) * np.inner(difference, difference)
        CP_remainder = self._nonnegative_or_error(
            self._CP[retained_i] - compactness - correction,
            max(
                abs(self._CP[retained_i]),
                abs(compactness),
                correction,
            ),
            "cluster compactness",
        )

        new_i = self._label_map.get_internal_label(new_label)
        if new_i != self._n_clusters:
            raise RuntimeError("New split label was not appended")

        self._n[retained_i] = n_remainder
        self._v[retained_i, :] = v_remainder
        self._CP[retained_i] = CP_remainder
        self._G[retained_i, :] = np.zeros(self._dim)
        self._n.append(count)
        self._v = np.vstack((self._v, centroid))
        self._CP.append(compactness)
        self._G = np.vstack((self._G, np.zeros(self._dim)))
        self._n_clusters += 1
        self._rebuild_after_operation()

    def remove(self, sample: np.ndarray, label: int) -> float:
        """
        Remove a sample from an initialized CVI.

        The caller is responsible for ensuring that the sample belongs to the
        supplied cluster label. If the sample is the cluster's final member,
        the empty cluster and its label are removed.

        Parameters
        ----------
        sample : numpy.ndarray
            One sample vector of features.
        label : int
            External label of the cluster containing the sample.

        Returns
        -------
        float
            The updated CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement removal.
        ValueError
            If the index is uninitialized, the label is unknown, the sample
            has the wrong shape, or the sample is inconsistent with the
            stored sufficient statistics.
        """

        self._require_operations()
        sample = self._validate_sample(sample)
        i_label = self._label_map.get_existing_label(label)
        self._remove(sample, label, i_label)
        self._evaluate()
        return self.criterion_value

    def merge(self, target_label: int, source_label: int) -> float:
        """
        Merge a source cluster into a target cluster.

        The target external label is retained and the source label is removed.

        Parameters
        ----------
        target_label : int
            External label of the cluster that remains after the merge.
        source_label : int
            External label of the cluster merged into the target.

        Returns
        -------
        float
            The updated CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement cluster merging.
        ValueError
            If the index is uninitialized, either label is unknown, or the
            two labels are equal.
        """

        self._require_operations()

        if target_label == source_label:
            raise ValueError("Merge requires two different cluster labels")

        target_i = self._label_map.get_existing_label(target_label)
        source_i = self._label_map.get_existing_label(source_label)
        self._merge(target_label, source_label, target_i, source_i)
        self._evaluate()
        return self.criterion_value

    def split(
        self,
        retained_label: int,
        new_label: int,
        count: int,
        centroid: np.ndarray,
        *,
        compactness: Optional[float] = None,
        covariance: Optional[np.ndarray] = None,
    ) -> float:
        """
        Split a tracked subset from an existing cluster.

        The existing external label is retained by the residual cluster. The
        supplied sufficient statistics are assigned to a new cluster with
        ``new_label``. The total sample count and global mean do not change.

        Parameters
        ----------
        retained_label : int
            External label of the cluster retaining the residual statistics.
        new_label : int
            Unused external label assigned to the split-off subset.
        count : int
            Number of samples in the split-off subset.
        centroid : numpy.ndarray
            Mean vector of the split-off subset.
        compactness : float, optional
            Sum of squared distances from the subset centroid. Required by
            compactness-based indices when ``count`` is greater than one.
        covariance : numpy.ndarray, optional
            Unregularized unbiased sample covariance of the subset. Required
            by rCIP when ``count`` is greater than one.

        Returns
        -------
        float
            The updated CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement cluster splitting.
        ValueError
            If the index is uninitialized, labels or statistics are invalid,
            or the supplied subset is inconsistent with the retained cluster.
        """

        self._require_operations()
        (
            retained_i,
            count,
            centroid,
            compactness,
            covariance,
        ) = self._validate_split_inputs(
            retained_label,
            new_label,
            count,
            centroid,
            compactness,
            covariance,
        )
        self._split(
            new_label,
            retained_i,
            count,
            centroid,
            compactness,
            covariance,
        )
        self._evaluate()
        return self.criterion_value

    def get_cvi(self, data: np.ndarray, label: Union[int, np.ndarray]) -> float:
        """
        Update the CVI and return its criterion value.

        Pass a one-dimensional sample and scalar integer label for an
        incremental update, or a two-dimensional batch and label vector for
        batch initialization. The object is mutated in both modes. A batch may
        be followed by incremental updates, but a second batch is not
        supported.

        Parameters
        ----------
        data : np.ndarray
            The sample(s) of features used for clustering.
        label : Union[int, np.ndarray]
            The label(s) prescribed to the sample(s) by the clustering algorithm.

        Returns
        -------
        float
            The CVI's criterion value.

        Raises
        ------
        ValueError
            If the input dimensionality is invalid, feature dimensionality
            changes after initialization, batch labels contain fewer than two
            distinct values, or a second batch update is requested.
        """

        # If we got 1D data, do a quick update
        if (data.ndim == 1):
            if self._is_setup and data.shape[0] != self._dim:
                raise ValueError(
                    f"Expected a sample with {self._dim} features, "
                    f"received {data.shape[0]}"
                )
            self._param_inc(data, label)

        # Otherwise, we got 2D data and do the correct update
        elif (data.ndim == 2):

            # If we haven't done a batch update yet
            if not self._is_setup:

                # Check that there are at least two unique labels
                if not len(np.unique(label)) > 1:
                    raise ValueError(
                        "Batch CVI mode requires at least two unique labels"
                    )

                # Do a batch update
                self._param_batch(data, label)

            # Otherwise, a second batch update was requested
            else:
                raise ValueError(
                    "Repeated batch updates are not supported"
                )

        # Otherwise, we got incorrectly dimensioned data
        else:

            # Error until some intelligent data sanitization is implemented
            raise ValueError(
                f"Please provide 1D or 2D numpy array, received ndim={data.ndim}"
            )

        # Regardless of path, evaluate and extract the criterion value
        self._evaluate()
        criterion_value = self.criterion_value

        # Return the criterion value
        return criterion_value


# --------------------------------------------------------------------------- #
# DECORATORS
# --------------------------------------------------------------------------- #


def _add_docs(docstring: str) -> Callable[[], None]:
    """
    A decorator for appending a string to the docstring of a function.

    Parameters
    ----------
    docstring : str
        The docstring that you want to append to the decorated function.
    """

    def dec(func):
        func.__doc__ = func.__doc__ + docstring
        return func

    return dec


# --------------------------------------------------------------------------- #
# DOCSTRINGS
# --------------------------------------------------------------------------- #

# This docstring documents the shared API for incremental setup
_setup_doc = (
    """
    Sets up the dimensions of the CVI based on the sample size.

    Parameters
    ----------
    sample : numpy.ndarray
        A sample vector of features.
    """
)

# This docstring documents the shared API for incremental parameter updates
_param_inc_doc = (
    """
    Parameters
    ----------
    sample : numpy.ndarray
        A sample row vector of features.
    label : int
        An integer identifier for the cluster. Labels need not be consecutive
        or zero-indexed.
    """
)

# This docstring documents the shared API for batch parameter updates
_param_batch_doc = (
    """
    Parameters
    ----------
    sample : numpy.ndarray
        A batch of samples; each row is a new sample of features.
    label : numpy.ndarray
        A vector of integer cluster identifiers. Labels need not be
        consecutive or zero-indexed.
    """
)

# This docstring documents the shared API for criterion value evaluation
_evaluate_doc = (
    """
    Updates the internal `criterion_value` parameter.
    """
)
