"""
Utilities that are common across all CVI objects.

# Authors
- Sasha Petrenko <sap625@mst.edu>
"""

import copy
import functools
import inspect
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Callable,
    ClassVar,
    Optional,
    Union,
)

# Custom imports
import numpy as np

from ..backends import BACKEND_NAMES, get_backend
from . import _batch, _trial

# --------------------------------------------------------------------------- #
# CLASSES
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CVIInfo:
    """Index metadata, independent of installation.

    ``backends`` lists numerical backend names, including optional backends
    whose dependencies may not be installed.
    """

    name: str
    name_short: str
    index_min: float
    index_max: float
    optimality: str
    backends: tuple[str, ...] = ("numpy",)


@dataclass(frozen=True)
class CVICapabilities:
    """Operations supported by one configured CVI instance.

    Capabilities describe the selected numerical and prototype backends. They
    do not indicate whether the object has been initialized or whether optional
    dependencies are installed. ``batch`` describes one-shot ``get_cvi``
    initialization; ``mini_batch`` describes cumulative ``update_batch``
    support (aggregation, or sequential updates for FuzzyART CONN),
    independently of sequential JAX ``update_many`` support. ``trial``
    describes scoring an operation without applying it (``commit=False``),
    and ``move`` describes :meth:`CVI.move`.
    """

    batch: bool
    incremental: bool
    merge: bool
    remove: bool
    split: bool
    mini_batch: bool = False
    trial: bool = False
    move: bool = False


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
    _uses_compactness_stats: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        """Record each object's constructor arguments for clone and reset."""
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:
            return

        @functools.wraps(init)
        def recording_init(self, *args, **init_kwargs):
            # The outermost constructor runs first and owns the configuration.
            if "_init_params" not in self.__dict__:
                self.__dict__["_init_params"] = (
                    copy.deepcopy(args), copy.deepcopy(init_kwargs),
                )
            init(self, *args, **init_kwargs)

        recording_init.__signature__ = inspect.signature(init)
        cls.__init__ = recording_init

    def __init__(self, *, backend="numpy", capacity=None):
        """
        CVI base class initialization method.

        Parameters
        ----------
        backend : {"numpy", "numba", "jax"}, default="numpy"
            Numerical implementation. Numba is optional and must be supported
            by the concrete index. The choice remains fixed through resets.
        capacity : int or None, default=None
            Maximum distinct clusters for optional fixed-capacity JAX streaming.
        """

        if backend in BACKEND_NAMES and backend not in self.info.backends:
            raise NotImplementedError(
                f"{type(self).__name__} does not support the {backend} backend"
            )
        if capacity is not None:
            if backend != "jax":
                raise ValueError("capacity is available only with backend='jax'")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                raise ValueError("capacity must be a positive integer")
        self._capacity = capacity
        self._stream_state = None
        self._backend = get_backend(backend)
        self._label_map = LabelMap()
        self._dim = 0
        self._n_samples = 0
        self._n = []                 # dim
        self._v = np.zeros([0, 0])   # n_clusters x dim
        self._CP = []                # dim
        self._G = np.zeros([0, 0])   # n_clusters x dim
        self._n_clusters = 0
        self.criterion_value = np.nan
        self._is_setup = False
        self._trial_memo = None

    @property
    def backend(self):
        """Selected numerical backend (fixed for this object's lifetime)."""
        return self._backend.name

    @property
    def capabilities(self) -> CVICapabilities:
        """Operations supported by this instance's configuration."""

        structural = self.backend != "jax"
        return CVICapabilities(
            batch=True,
            incremental=self.backend != "jax" or self.capacity is not None,
            merge=structural,
            remove=structural,
            split=structural,
            mini_batch=self.backend != "jax",
            trial=structural,
            move=structural,
        )

    # ----------------------------------------------------------------------- #
    # Read-only accessors and lifecycle
    # ----------------------------------------------------------------------- #

    @property
    def n_clusters(self) -> int:
        """Number of clusters currently tracked."""
        return len(self._label_map.map)

    @property
    def n_samples(self) -> int:
        """Number of samples currently represented."""
        return int(self._n_samples)

    @property
    def labels(self) -> tuple:
        """Known external cluster labels, in internal (first-seen) order."""
        mapping = self._label_map.map
        return tuple(sorted(mapping, key=mapping.__getitem__))

    @classmethod
    def is_better(cls, new: float, old: float, tol: float = 0.0) -> bool:
        """Return True if score ``new`` is at least as good as ``old``.

        Direction follows ``info.optimality``. Ties are accepted (``>=`` for
        maximized indices, ``<=`` for minimized ones), ``tol`` widens the
        acceptance band, and a NaN on either side counts as acceptance, since
        an undefined index cannot reject a candidate.
        """
        new, old = float(new), float(old)
        if np.isnan(new) or np.isnan(old):
            return True
        if cls.info.optimality == "max":
            return new >= old - tol
        return new <= old + tol

    def _constructor_params(self):
        """Constructor arguments, recovering them for older pickles."""
        params = self.__dict__.get("_init_params")
        if params is None:
            kwargs = {"backend": self.backend}
            if getattr(self, "_capacity", None) is not None:
                kwargs["capacity"] = self._capacity
            params = ((), kwargs)
        return params

    def clone(self):
        """Return a fresh, unfitted CVI with the same configuration."""
        args, kwargs = self._constructor_params()
        return type(self)(*copy.deepcopy(args), **copy.deepcopy(kwargs))

    def reset(self):
        """Clear all fitted data, keeping the configuration and backend.

        The object afterwards behaves like a newly constructed one, so a
        batch ``get_cvi`` can rebuild it.
        """
        backend = self._backend
        fresh = self.clone()
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        self._backend = backend
        return self

    @property
    def capacity(self):
        """Maximum distinct clusters for JAX streaming, or None for batch only."""
        return self._capacity

    @property
    def stream_state(self):
        """Immutable device state for JAX streaming; None until initialized."""
        return self._stream_state

    def update_many(self, data, labels, *, return_history=True):
        """Add a chunk to a fixed-capacity JAX stream, atomically on input errors.

        Return a NumPy score history, or a Python final score when
        return_history=False. Empty chunks are no-ops, including on new objects.
        The entire chunk must fit the remaining cluster capacity. This is an
        incremental scan, distinct from the one-time batch get_cvi operation.
        """
        if self.backend != "jax" or self.capacity is None:
            raise NotImplementedError("update_many requires JAX with capacity")
        mapping, updates, output = self._backend.advance(
            self, data, labels, return_history=return_history,
        )
        self.__dict__.update(updates)
        self._label_map.map = mapping
        self._forget_trial()
        return output

    def update_batch(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Append a labeled chunk and return the cumulative final score.

        Supported with NumPy (CONN requires ``model_type="Fuzzy"``), and by
        all indices supporting Numba with that backend. JAX is not supported.
        FuzzyART CONN overrides this method with sequential incremental
        updates; see :meth:`cvi.CONN.update_batch`. Other indices' groups are
        summarized in float64 and merged directly;
        intermediate per-sample scores are not computed. Existing batch,
        sample, and chunk state can all be continued with this method.

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
            Finite real observations, with at least one feature. The feature
            count must match an initialized index.
        labels : array-like, shape (n_samples,)
            Integer cluster identifiers. New labels are appended in first-seen
            order. Each row is a new observation; previous assignments remain.

        Returns
        -------
        float
            Score for all accumulated observations, or NaN without warning
            while undefined. Empty chunks return the existing score and leave
            even a fresh object uninitialized.

        Raises
        ------
        NotImplementedError
            If this index/backend does not support chunk updates.
        ValueError
            If inputs are invalid or summary statistics exceed float64 range.
            Failed updates leave the object unchanged.

        Notes
        -----
        Reduction order differs from one-shot batch and sequential updates;
        bitwise equality is not guaranteed. Unlike legacy batch reductions,
        float32 inputs are accumulated in float64. JAX ``update_many`` remains
        a separate sequential scan with optional per-sample score history.
        """
        if not self.capabilities.mini_batch:
            raise NotImplementedError(
                f"{type(self).__name__} does not support update_batch "
                f"with backend={self.backend!r}"
            )
        try:
            return _batch.update_batch(self, data, labels)
        finally:
            self._forget_trial()

    def _merge_batch_statistics(self, data, order, offsets, slots):
        """Merge a chunk into a staged object's sufficient statistics."""
        _batch.merge_default(self, data, order, offsets, slots)

    def __getstate__(self):
        """Serialize fitted state; the private trial memo is not persisted."""
        state = self.__dict__.copy()
        state.pop("_trial_memo", None)
        return state

    def __setstate__(self, state):
        """Treat objects serialized before backend selection as NumPy objects."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_capacity", None)
        self.__dict__.setdefault("_stream_state", None)
        self.__dict__["_trial_memo"] = None
        if "_backend" not in state:
            self._backend = get_backend("numpy")

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

    def _setup_batch_labels(self, labels: np.ndarray, *, return_inverse=False):
        """Populate first-seen labels, optionally returning dense sample labels.

        Integer labels are encoded by NumPy; only distinct labels enter Python.
        Other dtypes retain the dictionary path and its equality semantics.
        """

        self._label_map = LabelMap()
        labels = np.asarray(labels)
        if labels.dtype.kind in "biu":
            encoded = np.unique(labels, return_index=True,
                                return_inverse=return_inverse)
            unique, first = encoded[:2]
            first_seen = np.argsort(first)
            unique_labels = unique[first_seen].tolist()
            self._label_map.map = dict(zip(unique_labels, range(len(unique))))
            if return_inverse:
                remap = np.empty(len(unique), dtype=np.intp)
                remap[first_seen] = np.arange(len(unique))
                return unique_labels, remap[encoded[2]]
            return unique_labels

        unique_labels = []
        for label in labels:
            external_label = label.item() if hasattr(label, "item") else label
            if external_label not in self._label_map.map:
                self._label_map.get_internal_label(external_label)
                unique_labels.append(external_label)

        if return_inverse:
            dense_labels = np.fromiter(
                (self._label_map.map[label] for label in labels),
                dtype=np.intp, count=len(labels),
            )
            return unique_labels, dense_labels
        return unique_labels

    @staticmethod
    def _prepare_integer_batch_groups(labels: np.ndarray):
        """Group integer labels with one stable sort in first-seen order."""
        labels = np.asarray(labels)
        order_by_label = np.argsort(labels, kind="stable")
        if len(labels) == 0:
            return [], order_by_label, np.zeros(1, dtype=np.intp)

        sorted_labels = labels[order_by_label]
        starts = np.r_[
            0,
            np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1,
        ]
        stops = np.r_[starts[1:], len(labels)]
        first_seen = np.argsort(order_by_label[starts])
        unique_labels = sorted_labels[starts[first_seen]].tolist()
        counts = (stops - starts)[first_seen]
        order = np.concatenate([
            order_by_label[starts[group]:stops[group]]
            for group in first_seen
        ])
        offsets = np.r_[0, np.cumsum(counts)]
        return unique_labels, order, offsets

    def _setup_batch_groups(self, labels: np.ndarray):
        """Populate the label map and return stable first-seen groups."""
        prepared = getattr(self, "_prepared_batch_groups", None)
        if prepared is None:
            unique_labels, dense_labels = self._setup_batch_labels(
                labels, return_inverse=True,
            )
            order, offsets = self._backend.grouped_rows(
                dense_labels, len(unique_labels),
            )
            return unique_labels, order, offsets

        unique_labels, order, offsets = prepared
        self._label_map = LabelMap()
        self._label_map.map = dict(
            zip(unique_labels, range(len(unique_labels)))
        )
        return unique_labels, order, offsets

    def _setup_batch_statistics(self, data, labels, compactness=True):
        """Initialize shared batch state and return stable cluster grouping."""
        self._setup_batch(data)
        unique_labels, order, offsets = self._setup_batch_groups(labels)
        self._n_clusters = len(unique_labels)
        counts, self._v, squared_errors = self._backend.batch_statistics(
            data, order, offsets, compactness=compactness,
        )
        # Lists remain appendable by the existing incremental implementation.
        self._n = counts.tolist()
        if compactness:
            self._CP = list(squared_errors)
            self._G = np.zeros_like(self._v)
        return order, offsets

    @abstractmethod
    def _param_inc(self, sample: np.ndarray, label: int):
        """Plain committing addition (used when no matching trial exists).

        Trials and commits after a matching trial use the separate
        ``_plan_add``/``_apply_add`` path instead; tests keep the two in
        agreement.
        """
        raise NotImplementedError

    @abstractmethod
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def _evaluate(self):
        raise NotImplementedError

    def _require_operation(self, operation: str):
        """Validate that one structural operation is supported and available."""

        if not getattr(self.capabilities, operation):
            if self.backend != "jax":
                raise NotImplementedError(
                    f"{type(self).__name__} does not support {operation}"
                )
            if self.capacity is not None:
                raise NotImplementedError("JAX streaming does not support remove or merge")
            raise NotImplementedError("The jax backend currently supports batch only")

        if not self._is_setup:
            raise ValueError(
                "Remove, merge, split, and move require an initialized CVI"
            )

    def _validate_sample(self, sample: np.ndarray) -> np.ndarray:
        """Validate and normalize a sample used by a structural operation."""

        sample = np.asarray(sample, dtype=float)

        if sample.ndim != 1:
            raise ValueError("Expected a one-dimensional sample")

        if sample.shape[0] != self._dim:
            raise ValueError(
                f"Expected a sample with {self._dim} features, "
                f"received {sample.shape[0]}"
            )

        if not np.all(np.isfinite(sample)):
            raise ValueError("The sample must contain finite values")

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
        self.criterion_value = np.nan
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

    # ----------------------------------------------------------------------- #
    # Trial memo
    # ----------------------------------------------------------------------- #

    def _state_fingerprint(self):
        """Cheap guard that the object has not changed since a trial."""
        return (self._n_samples, self._n_clusters, len(self._label_map.map),
                self._dim, id(self._v))

    def _remember_trial(self, operation, arguments, payload):
        """Keep the newest trial's computed state for a matching commit."""
        self._trial_memo = (
            operation, arguments, self._state_fingerprint(), payload,
        )

    def _recall_trial(self, operation, arguments):
        """Return a matching trial's computed state, or None."""
        memo = self.__dict__.get("_trial_memo")
        if (
            memo is None
            or memo[0] != operation
            or memo[2] != self._state_fingerprint()
            or not _trial.same_arguments(memo[1], arguments)
        ):
            return None
        return memo[3]

    def _forget_trial(self):
        self.__dict__["_trial_memo"] = None

    def _snapshot(self):
        """Independent copy of the complete state, without the memo."""
        return copy.deepcopy(self)

    def _adopt(self, staged):
        """Replace this object's state with a staged copy's state."""
        backend = self._backend
        self.__dict__.clear()
        self.__dict__.update(staged.__dict__)
        self._backend = backend
        self._trial_memo = None

    def _require_trial(self):
        if not self.capabilities.trial:
            raise NotImplementedError(
                f"{type(self).__name__} does not support trial operations "
                f"(commit=False) with backend={self.backend!r}"
            )

    @staticmethod
    def _candidate_labels(label):
        """Normalize a scalar label or a 1-D array of candidate labels."""
        labels = np.asarray(label)
        if labels.ndim == 0:
            return [label], True
        if labels.ndim == 1:
            return labels.tolist(), False
        raise ValueError(
            "Candidate labels must be a scalar or a one-dimensional array"
        )

    @staticmethod
    def _require_scalar_label(label):
        if np.ndim(label) != 0:
            raise ValueError(
                "Committing requires a scalar label; pass commit=False to "
                "score several candidate labels"
            )

    # ----------------------------------------------------------------------- #
    # Incremental additions: a pure plan, then an apply step
    # ----------------------------------------------------------------------- #

    def _plan_add(self, sample, labels, *, score):
        """Compute, without mutation, the state of adding ``sample`` to each
        candidate label, and optionally the resulting scores."""
        plan = _trial.AddPlan(
            np.asarray(sample, dtype=np.float64),
            labels,
            self._label_map.map,
            fresh=self._n_samples == 0,
        )
        self._plan_add_statistics(plan)
        if not plan.fresh:
            self._plan_add_index(plan)
        if score:
            if plan.fresh:
                plan.scores = np.full(plan.n_slots, np.nan)
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    plan.scores = self._score_add(plan)
        return plan

    def _plan_centroids(self, plan, shift_form=False):
        """Counts and centroids for every plan slot."""
        x, slots, internal = plan.x, plan.existing_slots, plan.existing_labels
        if plan.n_slots == 1:
            # Committing plans one label; plain 1-D arithmetic is much
            # cheaper here and evaluates the same expressions elementwise.
            if len(slots):
                i = int(internal[0])
                n_old = self._n[i]
                n_new = n_old + 1
                v = self._v[i]
                if shift_form:
                    v_new = v + (x - v) / n_new
                else:
                    v_new = (1 - 1 / n_new) * v + (1 / n_new) * x
            else:
                n_old, n_new, v_new = 0, 1, x
            plan.n_old = np.asarray([n_old], dtype=np.int64)
            plan.n_new = np.asarray([n_new], dtype=np.int64)
            plan.v_new = v_new.reshape(1, -1).astype(np.float64, copy=True)
            return
        plan.n_old = np.zeros(plan.n_slots, dtype=np.int64)
        plan.v_new = np.empty((plan.n_slots, len(x)))
        plan.v_new[:] = x
        if len(slots):
            plan.n_old[slots] = [self._n[i] for i in internal.tolist()]
        plan.n_new = plan.n_old + 1
        if len(slots):
            v = self._v[internal]
            n_new = plan.n_new[slots][:, None]
            if shift_form:
                plan.v_new[slots] = v + (x - v) / n_new
            else:
                plan.v_new[slots] = (1 - 1 / n_new) * v + (1 / n_new) * x

    def _plan_add_statistics(self, plan):
        """Counts, centroids and, if used, compactness and the global mean."""
        self._plan_centroids(plan)
        if not self._uses_compactness_stats:
            return
        x, slots, internal = plan.x, plan.existing_slots, plan.existing_labels
        if plan.n_slots == 1:
            if len(slots):
                i = int(internal[0])
                n = plan.n_old[0]
                v_new = plan.v_new[0]
                G = self._G[i]
                delta_v = self._v[i] - v_new
                diff_x_v = x - v_new
                CP_new = (
                    self._CP[i]
                    + _trial.dot(diff_x_v, diff_x_v)
                    + n * _trial.dot(delta_v, delta_v)
                    + 2 * _trial.dot(delta_v, G)
                )
                G_new = G + diff_x_v + n * delta_v
            else:
                CP_new, G_new = 0.0, np.zeros(len(x))
            plan.CP_new = np.asarray([CP_new], dtype=np.float64)
            plan.G_new = G_new.reshape(1, -1)
        else:
            plan.CP_new = np.zeros(plan.n_slots)
            plan.G_new = np.zeros((plan.n_slots, len(x)))
            if len(slots):
                n = plan.n_old[slots]
                v_old = self._v[internal]
                v_new = plan.v_new[slots]
                G = self._G[internal]
                CP = np.asarray([self._CP[i] for i in internal.tolist()],
                                dtype=np.float64)
                delta_v = v_old - v_new
                diff_x_v = x - v_new
                plan.CP_new[slots] = (
                    CP
                    + _trial.row_dot(diff_x_v, diff_x_v)
                    + n * _trial.row_dot(delta_v, delta_v)
                    + 2 * _trial.row_dot(delta_v, G)
                )
                plan.G_new[slots] = G + diff_x_v + n[:, None] * delta_v
        n_samples_new = self._n_samples + 1
        plan.mu_new = (
            x.copy() if plan.fresh else
            (1 - 1 / n_samples_new) * self._mu + (1 / n_samples_new) * x
        )

    def _plan_add_index(self, plan):
        """Index-specific statistics of an initialized plan (default: none)."""

    def _score_add(self, plan):
        """Score every slot of an initialized plan."""
        raise NotImplementedError

    def _apply_add(self, plan, slot, label):
        """Apply one slot of a plan computed against the current state."""
        x = plan.x
        if self._n_samples == 0:
            self._setup(x)
        new = label not in self._label_map.map
        if new != (plan.cand[slot] < 0) or plan.fresh != (self._n_samples == 0):
            raise RuntimeError("The trial plan no longer matches the CVI state")
        i = self._label_map.get_internal_label(label)
        n_new = int(plan.n_new[slot])
        if new:
            self._n.append(n_new)
            self._v = np.vstack([self._v, plan.v_new[slot]])
        else:
            self._n[i] = n_new
            self._v[i, :] = plan.v_new[slot]
        if self._uses_compactness_stats:
            if new:
                self._CP.append(float(plan.CP_new[slot]))
                self._G = np.vstack([self._G, plan.G_new[slot]])
            else:
                self._CP[i] = float(plan.CP_new[slot])
                self._G[i, :] = plan.G_new[slot]
            if not plan.fresh:
                self._mu = plan.mu_new
        if new:
            self._n_clusters += 1
        self._apply_add_index(plan, slot, i, new)
        self._n_samples += 1

    def _apply_add_index(self, plan, slot, i_label, new):
        """Index-specific part of applying one plan slot."""

    def _trial_within(self, plan):
        """Within-group sum of squares for each slot."""
        total = sum(self._CP)
        within = np.full(plan.n_slots, total, dtype=float)
        existing, cand = plan.existing, plan.cand
        CP = np.asarray(self._CP, dtype=np.float64)
        within[existing] = total - CP[cand[existing]] + plan.CP_new[existing]
        return within

    def _trial_cluster_counts(self, plan):
        """Number of clusters after each slot's addition."""
        return self._n_clusters + (~plan.existing).astype(np.int64)

    def _candidate_rows(self, plan, row):
        """Stack ``row(slot)`` for every slot; zero existing self entries."""
        rows = np.empty((plan.n_slots, self._n_clusters))
        for slot in range(plan.n_slots):
            rows[slot] = row(slot)
        rows[plan.existing_slots, plan.existing_labels] = 0.0
        return rows

    def _apply_pairwise_row(self, matrix, row, i_label, new):
        """Replace or append one symmetric row of a pairwise matrix."""
        if new:
            k = matrix.shape[0]
            expanded = np.zeros((k + 1, k + 1))
            expanded[:k, :k] = matrix
            expanded[k, :k] = row
            expanded[:k, k] = row
            return expanded
        matrix[i_label, :] = row
        matrix[:, i_label] = row
        return matrix

    # ----------------------------------------------------------------------- #
    # Public operations
    # ----------------------------------------------------------------------- #

    def _removed_copy(self, sample, label, i_label):
        """A staged copy with one sample removed and its score evaluated."""
        staged = self._snapshot()
        staged._remove(sample, label, i_label)
        staged._evaluate()
        return staged

    def remove(
        self,
        sample: np.ndarray,
        label: int,
        *,
        commit: bool = True,
    ) -> float:
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
        commit : bool, default=True
            If False, return the score the removal would produce and leave
            the object unchanged. A following committed call with the same
            arguments reuses the computed state.

        Returns
        -------
        float
            The updated (or, for a trial, prospective) CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement removal.
        ValueError
            If the index is uninitialized, the label is unknown, the sample
            has the wrong shape, or the sample is inconsistent with the
            stored sufficient statistics.
        """

        if not commit:
            self._require_trial()
        self._require_operation("remove")
        sample = self._validate_sample(sample)
        i_label = self._label_map.get_existing_label(label)
        arguments = (sample, label)
        if not commit:
            staged = self._removed_copy(sample, label, i_label)
            self._remember_trial("remove", arguments, staged)
            return staged.criterion_value

        staged = self._recall_trial("remove", arguments)
        if staged is not None:
            self._adopt(staged)
            return self.criterion_value
        self._forget_trial()
        self._remove(sample, label, i_label)
        self._evaluate()
        return self.criterion_value

    def merge(
        self,
        target_label: int,
        source_label: int,
        *,
        commit: bool = True,
    ) -> float:
        """
        Merge a source cluster into a target cluster.

        The target external label is retained and the source label is removed.

        Parameters
        ----------
        target_label : int
            External label of the cluster that remains after the merge.
        source_label : int
            External label of the cluster merged into the target.
        commit : bool, default=True
            If False, return the score the merge would produce and leave the
            object unchanged. A following committed call with the same
            arguments reuses the computed state.

        Returns
        -------
        float
            The updated (or, for a trial, prospective) CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement cluster merging.
        ValueError
            If the index is uninitialized, either label is unknown, or the
            two labels are equal.
        """

        if not commit:
            self._require_trial()
        self._require_operation("merge")

        if target_label == source_label:
            raise ValueError("Merge requires two different cluster labels")

        target_i = self._label_map.get_existing_label(target_label)
        source_i = self._label_map.get_existing_label(source_label)
        arguments = (target_label, source_label)
        if not commit:
            staged = self._snapshot()
            staged._merge(target_label, source_label, target_i, source_i)
            staged._evaluate()
            self._remember_trial("merge", arguments, staged)
            return staged.criterion_value

        staged = self._recall_trial("merge", arguments)
        if staged is not None:
            self._adopt(staged)
            return self.criterion_value
        self._forget_trial()
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
        commit: bool = True,
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
        commit : bool, default=True
            If False, return the score the split would produce and leave the
            object unchanged. A following committed call with the same
            arguments reuses the computed state.

        Returns
        -------
        float
            The updated (or, for a trial, prospective) CVI criterion value.

        Raises
        ------
        NotImplementedError
            If this index does not implement cluster splitting.
        ValueError
            If the index is uninitialized, labels or statistics are invalid,
            or the supplied subset is inconsistent with the retained cluster.
        """

        if not commit:
            self._require_trial()
        self._require_operation("split")
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
        arguments = (
            retained_label, new_label, count, centroid, compactness, covariance,
        )
        target = self
        if not commit:
            target = self._snapshot()
        else:
            staged = self._recall_trial("split", arguments)
            if staged is not None:
                self._adopt(staged)
                return self.criterion_value
            self._forget_trial()
        target._split(
            new_label,
            retained_i,
            count,
            centroid,
            compactness,
            covariance,
        )
        target._evaluate()
        if not commit:
            self._remember_trial("split", arguments, target)
        return target.criterion_value

    def move(
        self,
        sample: np.ndarray,
        old_label: int,
        new_label,
        *,
        commit: bool = True,
    ):
        """
        Reassign one sample from its current cluster to another.

        A committed move leaves exactly the state of ``remove(sample,
        old_label)`` followed by ``get_cvi(sample, new_label)``, applied
        atomically: no intermediate state (such as an emptied singleton
        cluster being deleted) is ever visible, and a failure leaves the
        object unchanged. A label not yet seen creates a new cluster. Moving
        a sample to its own label is a no-op that returns the current score.

        Parameters
        ----------
        sample : numpy.ndarray
            One sample vector of features, currently counted in
            ``old_label``.
        old_label : int
            External label of the cluster that currently holds the sample.
        new_label : int or array-like of int
            Destination label. With ``commit=False`` this may be a 1-D array
            of candidate labels, which are all scored in one call.
        commit : bool, default=True
            If False, return the score(s) the move would produce and leave
            the object unchanged. A following committed move of the same
            sample from the same label to any scored candidate reuses the
            computed state.

        Returns
        -------
        float or numpy.ndarray
            The updated criterion value, or for a trial one prospective
            score per candidate label (a float for a scalar label).
        """

        if not commit:
            self._require_trial()
        self._require_operation("move")
        sample = self._validate_sample(sample)
        old_i = self._label_map.get_existing_label(old_label)
        arguments = (sample, old_label)

        if commit:
            self._require_scalar_label(new_label)
            if new_label == old_label:
                return self.criterion_value
            cached = self._recall_trial("move", arguments)
            self._forget_trial()
            if cached is not None and new_label in cached[1].slot_of:
                staged, plan = cached
                staged._apply_add(plan, plan.slot_of[new_label], new_label)
            else:
                staged = self._removed_copy(sample, old_label, old_i)
                staged._param_inc(sample, new_label)
            staged._evaluate()
            self._adopt(staged)
            return self.criterion_value

        labels, scalar = self._candidate_labels(new_label)
        staged = self._removed_copy(sample, old_label, old_i)
        plan = staged._plan_add(sample, labels, score=True)
        scores = plan.scores_for_candidates().astype(np.float64)
        unchanged = [label == old_label for label in labels]
        scores[np.asarray(unchanged, dtype=bool)] = self.criterion_value
        self._remember_trial("move", arguments, (staged, plan))
        return float(scores[0]) if scalar else scores

    def get_cvi(
        self,
        data: np.ndarray,
        label: Union[int, np.ndarray],
        *,
        commit: bool = True,
    ):
        """
        Update the CVI and return its criterion value.

        Pass a one-dimensional sample and scalar integer label for an
        incremental update, or a two-dimensional batch and label vector for
        batch initialization. The object is mutated in both modes. A batch may
        be followed by incremental updates, but a second batch is not
        supported. JAX supports incremental additions when capacity is provided;
        it rejects remove and merge.

        Parameters
        ----------
        data : np.ndarray
            The sample(s) of features used for clustering.
        label : Union[int, np.ndarray]
            The label(s) prescribed to the sample(s) by the clustering
            algorithm. A label not yet seen creates a new cluster. With a
            one-dimensional sample and ``commit=False``, this may be a 1-D
            array of candidate labels, each scored independently.
        commit : bool, default=True
            If False, return the score(s) the update would produce and leave
            the object unchanged. A following committed update of the same
            sample with any scored candidate label (or of the same batch)
            reuses the computed state instead of recomputing it.

        Returns
        -------
        float or numpy.ndarray
            The CVI's criterion value, or for a vectorized trial one score
            per candidate label. NaN while the index is undefined.

        Raises
        ------
        ValueError
            If the input dimensionality is invalid, feature dimensionality
            changes after initialization, batch labels contain fewer than two
            distinct values, a second batch update is requested, or several
            labels are passed to a committing update.

        Warns
        -----
        RuntimeWarning
            If the criterion is undefined after a batch evaluation. The
            returned value is still ``numpy.nan``.
        """

        if not commit:
            self._require_trial()
            return self._trial_get_cvi(data, label)

        if self.backend == "jax":
            data = np.asarray(data)
            if data.ndim == 1 and self.capacity is not None:
                labels = np.asarray(label)
                if labels.ndim != 0:
                    raise ValueError("Expected a scalar integer label")
                mapping, updates, output = self._backend.advance(
                    self, data[None, :], labels[None], return_history=False,
                    single=True,
                )
                self.__dict__.update(updates)
                self._label_map.map = mapping
                return output
            if data.ndim == 1:
                raise NotImplementedError(
                    "The jax backend currently supports batch only"
                )
            if self._is_setup:
                raise ValueError("Repeated batch updates are not supported")
            mapping, state = self._backend.initialize(
                data, label, self.info.name_short, capacity=self.capacity,
            )
            label_map = LabelMap()
            label_map.map = mapping
            self.__dict__.update(state)
            self._label_map = label_map
            criterion_value = self.criterion_value
            if data.ndim == 2 and np.isnan(criterion_value):
                warnings.warn(
                    f"{type(self).__name__} is undefined for the supplied batch; "
                    "returning nan.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return criterion_value

        data = np.asarray(data)

        # If we got 1D data, do a quick update
        if (data.ndim == 1):
            if self._is_setup and data.shape[0] != self._dim:
                raise ValueError(
                    f"Expected a sample with {self._dim} features, "
                    f"received {data.shape[0]}"
                )
            self._require_scalar_label(label)
            plan = None
            if self.__dict__.get("_trial_memo") is not None:
                plan = self._recall_trial(
                    "add", (np.asarray(data, dtype=np.float64),),
                )
                self._forget_trial()
            if plan is not None and label in plan.slot_of:
                self._apply_add(plan, plan.slot_of[label], label)
            else:
                self._param_inc(data, label)

        # Otherwise, we got 2D data and do the correct update
        elif (data.ndim == 2):

            # If we haven't done a batch update yet
            if not self._is_setup:
                staged = self._recall_trial(
                    "batch", (data, np.asarray(label)),
                )
                if staged is not None:
                    self._adopt(staged)
                    return self._warn_if_undefined_batch(
                        self.criterion_value, stacklevel=3,
                    )
                self._forget_trial()
                labels = np.asarray(label)
                prepared = None
                if labels.ndim == 1 and labels.dtype.kind in "biu":
                    prepared = self._prepare_integer_batch_groups(labels)
                    n_labels = len(prepared[0])
                else:
                    n_labels = len(np.unique(labels))
                if n_labels < 2:
                    raise ValueError(
                        "Batch CVI mode requires at least two unique labels"
                    )

                if prepared is None:
                    self._param_batch(data, label)
                else:
                    self._prepared_batch_groups = prepared
                    try:
                        self._param_batch(data, label)
                    finally:
                        del self._prepared_batch_groups

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

        if data.ndim == 2:
            self._warn_if_undefined_batch(criterion_value, stacklevel=3)

        # Return the criterion value
        return criterion_value

    def _warn_if_undefined_batch(self, criterion_value, stacklevel):
        if np.isnan(criterion_value):
            warnings.warn(
                f"{type(self).__name__} is undefined for the supplied batch; "
                "returning nan.",
                RuntimeWarning,
                stacklevel=stacklevel,
            )
        return criterion_value

    def _trial_get_cvi(self, data, label):
        """Score an addition (or a first batch) without changing the object."""
        data = np.asarray(data)
        if data.ndim == 1:
            if self._is_setup and data.shape[0] != self._dim:
                raise ValueError(
                    f"Expected a sample with {self._dim} features, "
                    f"received {data.shape[0]}"
                )
            labels, scalar = self._candidate_labels(label)
            plan = self._plan_add(data, labels, score=True)
            self._remember_trial("add", (plan.x,), plan)
            scores = plan.scores_for_candidates().astype(np.float64)
            return float(scores[0]) if scalar else scores
        if data.ndim == 2:
            if self._is_setup:
                raise ValueError("Repeated batch updates are not supported")
            staged = self.clone()
            staged._backend = self._backend
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                staged.get_cvi(data, label)
            self._remember_trial(
                "batch", (data.copy(), np.array(label, copy=True)), staged,
            )
            if caught:
                warnings.warn(caught[-1].message, caught[-1].category,
                              stacklevel=3)
            return staged.criterion_value
        raise ValueError(
            f"Please provide 1D or 2D numpy array, received ndim={data.ndim}"
        )


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
