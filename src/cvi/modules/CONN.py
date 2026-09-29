"""
Connectivity-based CONN Cluster Validity Index.

This implementation follows the CONN-style validity index for prototype-based
partitions. Unlike distance-only CVIs, CONN depends on the first and second
best matching prototypes associated with each sample.

Notes
-----
Incremental mode uses FuzzyART and assumes samples are already normalized to
the ART input domain, typically [0, 1]. Batch mode supports FuzzyART, KMeans,
and MiniBatchKMeans and can optionally normalize the full dataset before
processing.

The iCONN initialization rule is handled explicitly:
    1. The first sample creates the first ART category.
    2. The second sample forces creation of the second ART category by
       temporarily setting ART vigilance to 1.0.
    3. Subsequent samples use ordinary ART dynamics.

References
----------
1. E. Merényi, "A new cluster validity index for prototype based clustering algorithms based on inter-and intra-cluster density," 2007 International Joint Conference on Neural Networks, 2007.
2. K. Tasdemir and E. Merényi, "A validity index for prototype-based clustering of data sets with complex cluster structures," IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 41, no. 4, pp. 1039-1053, 2011.
3. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental cluster validity indices for online learning of hard partitions: Extensions and comparative study," IEEE Access, vol. 8, pp. 22025-22047, 2020.
"""

# Standard library imports
from collections import defaultdict
from typing import Dict, Literal, Optional, Sequence, Union
import numbers

# Third-party imports
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# Local imports
from . import _base


def __getattr__(name):
    """Resolve legacy ART adapter paths lazily, including existing pickles."""
    if name in ("_CONNFuzzyART", "_CONNSimpleARTMAP"):
        from . import _conn_art
        return getattr(_conn_art, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _GrowingSquareArray:
    """
    Utility for square arrays whose size is determined online.
    """

    def __init__(self, dtype=float):
        self.array = np.zeros((0, 0), dtype=dtype)

    def _ensure_size(self, i: int, j: int):
        size = max(i + 1, j + 1)

        if size > self.array.shape[0]:
            new_array = np.zeros((size, size), dtype=self.array.dtype)

            if self.array.size > 0:
                old_size = self.array.shape[0]
                new_array[:old_size, :old_size] = self.array

            self.array = new_array

    def __getitem__(self, idx):
        i, j = idx

        # Allow NumPy advanced indexing without resizing.
        if (
            not isinstance(i, numbers.Integral)
            or not isinstance(j, numbers.Integral)
        ):
            return self.array[idx]

        self._ensure_size(i, j)
        return self.array[i, j]

    def __setitem__(self, idx, value):
        i, j = idx
        self._ensure_size(i, j)
        self.array[i, j] = value

    def increment(self, i: int, j: int, value=1):
        self._ensure_size(i, j)
        self.array[i, j] += value

    def asarray(self):
        return self.array.copy()

    def __repr__(self):
        return repr(self.array)


class _GrowingArray1D:
    """
    Utility for one-dimensional arrays whose size is determined online.
    """

    def __init__(self, dtype=float):
        self.array = np.zeros(0, dtype=dtype)

    def _ensure_size(self, i: int):
        if i >= self.array.size:
            new_array = np.zeros(i + 1, dtype=self.array.dtype)
            new_array[:self.array.size] = self.array
            self.array = new_array

    def __getitem__(self, i: int):
        self._ensure_size(i)
        return self.array[i]

    def __setitem__(self, i: int, value):
        self._ensure_size(i)
        self.array[i] = value

    def increment(self, i: int, value=1):
        self._ensure_size(i)
        self.array[i] += value

    def asarray(self):
        return self.array.copy()

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def __repr__(self):
        return repr(self.array)


class CONN(_base.CVI):
    """
    CONN Cluster Validity Index.

    Incremental mode uses a FuzzyART/SimpleARTMAP model. Batch mode can use
    FuzzyART or fit class-owned KMeans/MiniBatchKMeans prototypes.

    References
    ----------
    1. E. Merényi, "A new cluster validity index for prototype based clustering algorithms based on inter-and intra-cluster density," 2007 International Joint Conference on Neural Networks, 2007.
    2. K. Tasdemir and E. Merényi, "A validity index for prototype-based clustering of data sets with complex cluster structures," IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 41, no. 4, pp. 1039-1053, 2011.
    3. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental cluster validity indices for online learning of hard partitions: Extensions and comparative study," IEEE Access, vol. 8, pp. 22025-22047, 2020.
    """

    info = _base.CVIInfo(
        name="Connectivity",
        name_short="CONN",
        index_min=0.0,
        index_max=1.0,
        optimality="max",
        batch=True,
        incremental=True,
        merge=True,
        remove=False,
        split=True,
        backends=("numpy",),
    )

    _DEFAULT_RHO = 0.9
    _DEFAULT_ALPHA = 1e-10
    _DEFAULT_BETA = 1.0
    _DEFAULT_MATCH_TRACKING = "MT+"
    _DEFAULT_KMEANS_K = 8

    def __init__(
        self,
        rho: Optional[float] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        match_tracking: Optional[str] = None,
        normalize_batch: bool = True,
        check_incremental_normalized: bool = True,
        model_type: Literal["Fuzzy", "KMeans", "MiniBatchKMeans"] = (
            "MiniBatchKMeans"
        ),
        kmeans_k: Optional[Union[int, Dict[int, int]]] = None,
        kmeans_kwargs: Optional[dict] = None,
        *,
        backend: str = "numpy",
    ):
        """
        CONN initialization routine.

        Parameters
        ----------
        rho : float, optional
            FuzzyART vigilance parameter. Used only by ``model_type="Fuzzy"``
            and defaults to 0.9 for that model.
        alpha : float, optional
            FuzzyART choice parameter. Used only by ``model_type="Fuzzy"``
            and defaults to 1e-10 for that model.
        beta : float, optional
            FuzzyART learning rate. Used only by ``model_type="Fuzzy"`` and
            defaults to 1.0 for that model.
        match_tracking : str, optional
            Match-tracking mode passed to SimpleARTMAP. Used only by
            ``model_type="Fuzzy"`` and defaults to ``"MT+"`` for that model.
        normalize_batch : bool, default=True
            If True, batch data are min-max normalized before prototype
            fitting. Incremental data are not normalized online.
        check_incremental_normalized : bool, default=True
            If True, incremental samples are checked to ensure values lie in
            [0, 1].
        model_type : {"Fuzzy", "KMeans", "MiniBatchKMeans"}, default="MiniBatchKMeans"
            Prototype backend. KMeans backends support batch mode only.
        kmeans_k : int or dict[int, int], optional
            Number of KMeans prototypes per input label. Used only by a KMeans
            model and defaults to 8 for those models. Dictionary values are
            keyed by the original input labels. Counts are capped at the number
            of samples carrying each label.
        kmeans_kwargs : dict, optional
            Keyword arguments forwarded to the selected scikit-learn KMeans
            estimator. Used only by a KMeans model. ``n_clusters`` must be
            configured through ``kmeans_k``.
        backend : {"numpy"}, default="numpy"
            Numerical backend. CONN currently supports NumPy only; this is
            separate from the prototype algorithm selected by ``model_type``.
        """

        super().__init__(backend=backend)

        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.match_tracking = match_tracking
        self.normalize_batch = normalize_batch
        self.check_incremental_normalized = check_incremental_normalized
        self.model_type = model_type
        self.kmeans_k = kmeans_k
        self.kmeans_kwargs = kmeans_kwargs

        self._validate_backend_params()

        self._data_min = None
        self._data_max = None

        self._init_conn_state()

    def _init_conn_state(self):
        """
        Initialize or reset CONN state without constructing a prototype model.
        """

        self._artmap = None

        # Prototype-level matrices.
        self._CADJ = _GrowingSquareArray(dtype=float)
        self._CONN = _GrowingSquareArray(dtype=float)

        # Label-level arrays/matrices.
        self._INTRA = _GrowingArray1D(dtype=float)
        self._INTER = _GrowingSquareArray(dtype=float)

        self._intra_conn = 0.0
        self._inter_conn = 0.0

        # Internal label -> set of prototypes assigned to that label.
        self._rev_map = defaultdict(set)

        # Number of samples per internal label.
        self._cluster_cardinality = _GrowingArray1D(dtype=float)

        # Batch centroid-backend state.
        self._kmeans_models = None
        self._cluster_centers = None
        self._prototype_label_map = None
        self._prototype_cardinality = None

    def _ensure_artmap(self):
        """Construct and return the ART prototype model on first use."""

        if self.model_type != "Fuzzy":
            raise RuntimeError("ART initialization requires model_type='Fuzzy'.")

        if self._artmap is None:
            try:
                from ._conn_art import _CONNFuzzyART, _CONNSimpleARTMAP
            except ModuleNotFoundError as error:
                if (error.name or "").split(".")[0] != "artlib":
                    raise
                raise ImportError(
                    "The Fuzzy CONN model requires the optional ART dependency; "
                    'install it with pip install "cvi[art]".'
                ) from error

            module_a = _CONNFuzzyART(
                rho=self.rho,
                alpha=self.alpha,
                beta=self.beta,
            )
            self._artmap = _CONNSimpleARTMAP(module_a)

        return self._artmap

    def _ensure_kmeans_state(self):
        """Initialize and return the KMeans prototype state on first use."""

        if self.model_type == "Fuzzy":
            raise RuntimeError("KMeans initialization requires a KMeans model type.")

        if self._kmeans_models is None:
            self._kmeans_models = {}
            self._cluster_centers = np.zeros((0, self._dim), dtype=float)
            self._prototype_label_map = {}

        return self._kmeans_models

    @staticmethod
    def _validate_positive_int(value, name: str) -> int:
        """Validate and return a strictly positive integer parameter."""

        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} must be a positive integer.")

        value = int(value)

        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

        return value

    def _validate_backend_params(self):
        """Resolve defaults and validate only the selected prototype model."""

        valid_model_types = {"Fuzzy", "KMeans", "MiniBatchKMeans"}

        if self.model_type not in valid_model_types:
            raise ValueError(
                "model_type must be one of "
                "{'Fuzzy', 'KMeans', 'MiniBatchKMeans'}."
            )

        if self.model_type == "Fuzzy":
            if self.rho is None:
                self.rho = self._DEFAULT_RHO
            if self.alpha is None:
                self.alpha = self._DEFAULT_ALPHA
            if self.beta is None:
                self.beta = self._DEFAULT_BETA
            if self.match_tracking is None:
                self.match_tracking = self._DEFAULT_MATCH_TRACKING
            return

        if self.kmeans_k is None:
            self.kmeans_k = self._DEFAULT_KMEANS_K

        if isinstance(self.kmeans_k, dict):
            for label, value in self.kmeans_k.items():
                if (
                    isinstance(label, bool)
                    or not isinstance(label, numbers.Integral)
                ):
                    raise ValueError("kmeans_k dictionary keys must be integers.")

                self._validate_positive_int(
                    value,
                    f"kmeans_k for label {int(label)}",
                )
        else:
            self._validate_positive_int(self.kmeans_k, "kmeans_k")

        if self.kmeans_kwargs is not None:
            if not isinstance(self.kmeans_kwargs, dict):
                raise ValueError("kmeans_kwargs must be a dictionary or None.")

            if "n_clusters" in self.kmeans_kwargs:
                raise ValueError(
                    "Configure n_clusters through kmeans_k, not kmeans_kwargs."
                )

            self.kmeans_kwargs = dict(self.kmeans_kwargs)

    @_base._add_docs(_base._setup_doc)
    def _setup(self, sample: np.ndarray):
        """
        CONN setup routine.
        """

        super()._setup(sample)

    def _normalize_batch_data(self, data: np.ndarray) -> np.ndarray:
        """
        Min-max normalize batch data featurewise.

        Constant-valued features are mapped to zero.
        """

        data = np.asarray(data, dtype=float)

        self._data_min = np.min(data, axis=0)
        self._data_max = np.max(data, axis=0)

        denom = self._data_max - self._data_min
        denom[denom == 0.0] = 1.0

        return (data - self._data_min) / denom

    def _get_kmeans_k(self, label: int, n_samples: int) -> int:
        """Resolve and cap the prototype count for one external label."""

        if isinstance(self.kmeans_k, dict):
            if label not in self.kmeans_k:
                raise ValueError(
                    f"kmeans_k is missing a value for label {label}."
                )

            requested = self.kmeans_k[label]
        else:
            requested = self.kmeans_k

        requested = self._validate_positive_int(
            requested,
            f"kmeans_k for label {label}",
        )

        return min(requested, n_samples)

    def _fit_centroid_backend(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """Fit class-owned KMeans prototypes and populate their label maps."""

        self._ensure_kmeans_state()

        estimator_type = (
            KMeans if self.model_type == "KMeans" else MiniBatchKMeans
        )
        model_kwargs = self.kmeans_kwargs or {}
        center_blocks = []
        next_prototype = 0
        ordered_labels = list(dict.fromkeys(int(label) for label in labels))

        if isinstance(self.kmeans_k, dict):
            missing_labels = [
                label for label in ordered_labels if label not in self.kmeans_k
            ]

            if missing_labels:
                raise ValueError(
                    "kmeans_k is missing values for labels "
                    f"{missing_labels}."
                )

        for label in ordered_labels:
            rows = np.flatnonzero(labels == label)
            i_label = self._label_map.get_internal_label(label)
            n_clusters = self._get_kmeans_k(label, len(rows))

            model = estimator_type(
                n_clusters=n_clusters,
                **model_kwargs,
            ).fit(data[rows])
            centers = np.asarray(model.cluster_centers_, dtype=float)
            prototype_ids = range(
                next_prototype,
                next_prototype + len(centers),
            )

            self._kmeans_models[label] = model
            self._rev_map[i_label].update(prototype_ids)

            for prototype_id in prototype_ids:
                self._prototype_label_map[prototype_id] = i_label

            center_blocks.append(centers)
            next_prototype += len(centers)

        self._cluster_centers = np.vstack(center_blocks)
        self._prototype_cardinality = np.zeros(
            len(self._cluster_centers), dtype=int
        )

        last_prototype = len(self._cluster_centers) - 1
        self._CADJ._ensure_size(last_prototype, last_prototype)
        self._CONN._ensure_size(last_prototype, last_prototype)

    def _update_conn_from_centroids(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ):
        """Compute batch CONN statistics from fixed centroid prototypes."""

        self._fit_centroid_backend(data, labels)

        for sample, label in zip(data, labels):
            i_label = self._label_map.get_internal_label(int(label))
            own_prototypes = np.asarray(
                sorted(self._rev_map[i_label]),
                dtype=int,
            )
            distances = np.sum(
                (self._cluster_centers - sample) ** 2,
                axis=1,
            )

            own_distances = distances[own_prototypes]
            bmu1 = int(own_prototypes[np.argmin(own_distances)])

            distances[bmu1] = np.inf
            bmu2 = int(np.argmin(distances))

            self._finish_conn_update(
                i_label,
                bmu1,
                bmu2,
                prototype_label_map=self._prototype_label_map,
                update_metric=False,
            )
            self._prototype_cardinality[bmu1] += 1
            self._n_samples += 1

        self._sync_base_cluster_count()

        # Recompute every row from the completed adjacency matrices so the
        # fixed-prototype batch result does not depend on update order.
        for i_label in sorted(self._rev_map):
            self._update_metric(i_label, i_label)

    def _check_sample_normalized(self, sample: np.ndarray):
        """
        Validate that an incremental sample is in the ART input domain.
        """

        if not self.check_incremental_normalized:
            return

        if np.any(sample < 0.0) or np.any(sample > 1.0):
            raise ValueError(
                "Incremental CONN assumes samples are already normalized "
                "to the ART input domain [0, 1]. For offline evaluation, "
                "use batch mode with normalize_batch=True."
            )

    def _set_module_rho(self, rho: float):
        """
        Set FuzzyART vigilance in a way that is robust to artlib storing rho
        both as an attribute and inside the params dictionary.
        """

        self._artmap.module_a.rho = rho

        if hasattr(self._artmap.module_a, "params"):
            self._artmap.module_a.params["rho"] = rho

    def _get_module_rho(self) -> float:
        """
        Get the current FuzzyART vigilance.
        """

        if hasattr(self._artmap.module_a, "rho"):
            return self._artmap.module_a.rho

        return self._artmap.module_a.params["rho"]

    def _force_second_category(self, sample_cc: np.ndarray, i_label: int):
        """
        Force creation of the second ART category for iCONN initialization.

        This temporarily sets ART vigilance to 1.0, performs one ARTMAP update,
        and then restores the original vigilance.
        """

        old_rho = self._get_module_rho()

        try:
            self._set_module_rho(1.0)

            self._artmap = self._artmap.partial_fit(
                np.asarray([sample_cc]),
                np.asarray([i_label]),
                match_tracking=self.match_tracking,
            )

        finally:
            self._set_module_rho(old_rho)

        if len(self._artmap.module_a.W) < 2:
            raise RuntimeError(
                "Failed to force creation of the second ART category. "
                "This can occur if the second sample perfectly resonates "
                "with the first sample at rho=1.0."
            )

    def _sync_base_cluster_count(self):
        """
        Synchronize the base CVI cluster counter with the internal label map.
        """

        self._n_clusters = len(self._label_map.map)

    def _require_prototype_operation(self):
        """Require an initialized model for prototype reassignment."""

        if not self._is_setup:
            raise ValueError("Merge and split require an initialized CVI")
        if self.model_type == "Fuzzy" and not hasattr(
            self._artmap, "move_A_prototype"
        ):
            raise ImportError(
                "CONN prototype merge and split require ARTlib>=0.1.12"
            )

    def _prototype_mapping(self) -> dict:
        """Return the selected backend's prototype-to-internal-label map."""

        if self.model_type == "Fuzzy":
            return self._artmap.map
        return self._prototype_label_map

    def get_prototype_ids(self, label: int) -> tuple[int, ...]:
        """Return global prototype IDs assigned to an external label."""

        if not self._is_setup:
            raise ValueError("Prototype IDs require an initialized CVI")
        internal_label = self._label_map.get_existing_label(label)
        prototype_map = self._prototype_mapping()
        return tuple(
            sorted(
                int(prototype_id)
                for prototype_id, assigned_label in prototype_map.items()
                if int(assigned_label) == internal_label
            )
        )

    def _rebuild_after_operation(self):
        """Recompute label-level CONN statistics from prototype assignments."""

        n_clusters = len(self._label_map.map)
        n_prototypes = (
            len(self._artmap.module_a.W)
            if self.model_type == "Fuzzy"
            else len(self._cluster_centers)
        )
        prototype_map = self._prototype_mapping()
        prototype_labels = np.asarray(
            [int(prototype_map[idx]) for idx in range(n_prototypes)],
            dtype=int,
        )
        if self.model_type == "Fuzzy":
            a_labels = np.asarray(self._artmap.module_a.labels_, dtype=int)
            if np.any(a_labels < 0) or np.any(a_labels >= n_prototypes):
                raise RuntimeError(
                    "ART sample labels refer to an unknown prototype"
                )
            cardinality = np.bincount(
                prototype_labels[a_labels], minlength=n_clusters
            ).astype(float)
        else:
            if len(self._prototype_cardinality) != n_prototypes:
                raise RuntimeError("KMeans prototype counts are incomplete")
            if self._prototype_cardinality.sum() != self._n_samples:
                raise RuntimeError(
                    "KMeans prototype counts disagree with samples"
                )
            cardinality = np.bincount(
                prototype_labels,
                weights=self._prototype_cardinality,
                minlength=n_clusters,
            ).astype(float)

        rev_map = defaultdict(set)
        for prototype_id, label in enumerate(prototype_labels):
            rev_map[int(label)].add(prototype_id)
        if set(rev_map) != set(range(n_clusters)):
            raise RuntimeError("Prototype labels disagree with CONN clusters")

        self._rev_map = rev_map
        self._cluster_cardinality = _GrowingArray1D(dtype=float)
        self._cluster_cardinality.array = cardinality

        intra = np.zeros(n_clusters, dtype=float)
        inter = np.zeros((n_clusters, n_clusters), dtype=float)
        for label, prototypes in rev_map.items():
            if cardinality[label] > 0:
                ids = np.asarray(sorted(prototypes), dtype=int)
                intra[label] = (
                    self._CADJ[np.ix_(ids, ids)].sum() / cardinality[label]
                )
            for other in range(n_clusters):
                if other != label:
                    inter[label, other] = self._calc_inter(label, other)

        self._INTRA = _GrowingArray1D(dtype=float)
        self._INTRA.array = intra
        self._INTER = _GrowingSquareArray(dtype=float)
        self._INTER.array = inter
        self._intra_conn = float(np.mean(intra))
        self._inter_conn = (
            float(np.mean(np.max(inter, axis=1))) if n_clusters > 1 else 0.0
        )
        self.criterion_value = self._intra_conn * (1.0 - self._inter_conn)
        self._sync_base_cluster_count()

    def merge(self, target_label: int, source_label: int) -> float:
        """Move every source prototype into the target cluster.

        The target external label is retained. Prototype-level connectivity
        and prototype parameters remain unchanged; label metrics are rebuilt.
        """

        self._require_prototype_operation()
        if target_label == source_label:
            raise ValueError("Merge requires two different cluster labels")
        target_i = self._label_map.get_existing_label(target_label)
        source_i = self._label_map.get_existing_label(source_label)
        prototype_map = self._prototype_mapping()
        source_prototypes = sorted(
            prototype_id
            for prototype_id, label in prototype_map.items()
            if int(label) == source_i
        )

        if self.model_type == "Fuzzy":
            for prototype_id in source_prototypes:
                self._artmap.move_A_prototype(source_i, prototype_id, target_i)

            # LabelMap compacts internal labels when the source is removed.
            # Apply the same shift through ARTlib to update its sample labels.
            for prototype_id in sorted(self._artmap.map):
                current_i = int(self._artmap.map[prototype_id])
                if current_i > source_i:
                    self._artmap.move_A_prototype(
                        current_i, prototype_id, current_i - 1
                    )
        else:
            remapped = {}
            for prototype_id, label in prototype_map.items():
                merged_label = target_i if label == source_i else label
                remapped[prototype_id] = (
                    merged_label - (merged_label > source_i)
                )
            self._prototype_label_map = remapped

        self._label_map.remove_label(source_label)
        self._rebuild_after_operation()
        return self.criterion_value

    def split(
        self,
        retained_label: int,
        new_label: int,
        prototype_ids: Sequence[int],
    ) -> float:
        """Move a proper subset of a cluster's prototypes to a new label.

        ``prototype_ids`` are global indices into the selected backend's
        prototype list. Samples assigned to moved prototypes follow their
        new cluster label.
        """

        self._require_prototype_operation()
        retained_i = self._label_map.get_existing_label(retained_label)
        if new_label in self._label_map.map:
            raise ValueError(
                f"Split requires an unused new cluster label: {new_label}"
            )

        try:
            prototype_ids = tuple(prototype_ids)
        except TypeError as error:
            raise ValueError(
                "Split prototype IDs must be a sequence"
            ) from error
        if not prototype_ids:
            raise ValueError("Split requires at least one prototype")
        if any(
            isinstance(idx, (bool, np.bool_))
            or not isinstance(idx, numbers.Integral)
            for idx in prototype_ids
        ):
            raise ValueError("Split prototype IDs must be integers")
        prototype_ids = tuple(int(idx) for idx in prototype_ids)
        if len(set(prototype_ids)) != len(prototype_ids):
            raise ValueError("Split prototype IDs must be unique")

        source_prototypes = {
            int(prototype_id)
            for prototype_id, label in self._prototype_mapping().items()
            if int(label) == retained_i
        }
        if not set(prototype_ids) <= source_prototypes:
            raise ValueError(
                "Split prototypes must belong to the retained label"
            )
        if len(prototype_ids) == len(source_prototypes):
            raise ValueError(
                "Split must leave a prototype in the retained label"
            )
        if self.model_type != "Fuzzy":
            moved_samples = int(
                self._prototype_cardinality[list(prototype_ids)].sum()
            )
            retained_samples = int(
                self._cluster_cardinality[retained_i] - moved_samples
            )
            if moved_samples == 0 or retained_samples == 0:
                raise ValueError(
                    "Split must leave assigned samples in both clusters"
                )

        new_i = len(self._label_map.map)
        if self.model_type == "Fuzzy":
            for prototype_id in prototype_ids:
                self._artmap.move_A_prototype(retained_i, prototype_id, new_i)
        else:
            for prototype_id in prototype_ids:
                self._prototype_label_map[prototype_id] = new_i

        self._label_map.get_internal_label(new_label)
        self._rebuild_after_operation()
        return self.criterion_value

    def _calc_inter(self, i_label: int, j_label: int) -> float:
        """
        Compute directed INTER connectivity from one internal label to another.
        """

        if i_label not in self._rev_map or j_label not in self._rev_map:
            return 0.0

        s1 = np.asarray(sorted(self._rev_map[i_label]), dtype=int)
        s2 = np.asarray(sorted(self._rev_map[j_label]), dtype=int)

        if s1.size == 0 or s2.size == 0:
            return 0.0

        cadj_sub = self._CADJ[np.ix_(s1, s2)]
        conn_sub = self._CONN[np.ix_(s1, s2)]

        inter_numer = conn_sub.sum()

        valid_rows = np.any(cadj_sub > 0, axis=1)
        inter_denom = conn_sub[valid_rows, :].sum()

        if inter_denom == 0.0:
            return 0.0

        return float(inter_numer / inter_denom)

    def _update_metric(self, y: int, y2: int):
        """
        Update INTRA, INTER, and the final CONN criterion value.
        """

        categories_y = np.asarray(sorted(self._rev_map[y]), dtype=int)

        if categories_y.size == 0 or self._cluster_cardinality[y] == 0:
            self._INTRA[y] = 0.0
        else:
            intra_numer = self._CADJ[np.ix_(categories_y, categories_y)].sum()
            self._INTRA[y] = intra_numer / self._cluster_cardinality[y]

        active_labels = sorted(self._rev_map.keys())

        if len(active_labels) == 0:
            self._intra_conn = 0.0
        else:
            self._intra_conn = float(
                sum(self._INTRA[label] for label in active_labels)
                / len(active_labels)
            )

        if y != y2:
            self._INTER[y, y2] = self._calc_inter(y, y2)
            self._INTER[y2, y] = self._calc_inter(y2, y)
        else:
            for label in active_labels:
                if label != y:
                    self._INTER[y, label] = self._calc_inter(y, label)

        if len(active_labels) < 2:
            self._inter_conn = 0.0
        else:
            row_maxes = []

            for label in active_labels:
                off_diag_values = [
                    self._INTER[label, other]
                    for other in active_labels
                    if other != label
                ]

                if off_diag_values:
                    row_maxes.append(max(off_diag_values))
                else:
                    row_maxes.append(0.0)

            self._inter_conn = float(np.mean(row_maxes))

        self.criterion_value = self._intra_conn * (1.0 - self._inter_conn)

    def _finish_conn_update(
        self,
        i_label: int,
        bmu1: int,
        bmu2: int,
        prototype_label_map: Optional[dict] = None,
        update_metric: bool = True,
    ):
        """
        Finish the CONN bookkeeping once BMU1 and BMU2 are known.
        """

        if prototype_label_map is None:
            prototype_label_map = self._ensure_artmap().map

        self._rev_map[i_label].add(bmu1)
        self._cluster_cardinality.increment(i_label, 1)

        self._CADJ.increment(bmu1, bmu2, 1)

        # CONN is the symmetrized co-adjacency.
        conn_value = self._CADJ[bmu1, bmu2] + self._CADJ[bmu2, bmu1]
        self._CONN[bmu1, bmu2] = conn_value
        self._CONN[bmu2, bmu1] = conn_value

        if bmu1 not in prototype_label_map:
            raise RuntimeError("BMU1 is missing from the prototype label map.")

        if int(prototype_label_map[bmu1]) != i_label:
            raise RuntimeError(
                "Internal prototype mapping disagrees with the provided label."
            )

        if not update_metric:
            return

        if bmu2 in prototype_label_map:
            y2 = int(prototype_label_map[bmu2])
        else:
            y2 = i_label

        self._update_metric(i_label, y2)

    def _update_conn_from_sample(self, sample: np.ndarray, label: int):
        """
        Update ART state and CONN sufficient statistics using one sample.
        """

        # Resolve the optional dependency before mutating any index state.
        self._ensure_artmap()

        sample = np.asarray(sample, dtype=float)
        i_label = self._label_map.get_internal_label(int(label))

        if not self._is_setup:
            self._setup(sample)

        self._check_sample_normalized(sample)

        from artlib.common.utils import complement_code

        # ART operates on complement-coded samples.
        sample_cc = complement_code(np.asarray([sample]))[0]

        # First sample:
        # Learn normally. CONN/iCONN is not yet defined because there is no
        # second ART category.
        if self._n_samples == 0:
            self._artmap = self._artmap.partial_fit(
                np.asarray([sample_cc]),
                np.asarray([i_label]),
                match_tracking=self.match_tracking,
            )

            bmu1 = int(self._artmap.module_a.labels_[-1])
            self._rev_map[i_label].add(bmu1)
            self._cluster_cardinality.increment(i_label, 1)

            self._n_samples += 1
            self._sync_base_cluster_count()

            self.criterion_value = np.nan
            return

        # Second sample:
        # Force creation of the second ART category according to the iCONN
        # initialization rule.
        if self._n_samples == 1 and len(self._artmap.module_a.W) == 1:
            self._force_second_category(sample_cc, i_label)

            bmu1 = int(self._artmap.module_a.labels_[-1])

            # With exactly two categories, the other category is BMU2.
            bmu2 = 1 - bmu1

            self._finish_conn_update(i_label, bmu1, bmu2)

            self._n_samples += 1
            self._sync_base_cluster_count()
            return

        # Normal update after iCONN initialization.
        self._artmap = self._artmap.partial_fit(
            np.asarray([sample_cc]),
            np.asarray([i_label]),
            match_tracking=self.match_tracking,
        )

        bmu1 = int(self._artmap.module_a.labels_[-1])

        c1, c2 = self._artmap.module_a.step_pred_first_and_second(sample_cc)
        bmu2 = c2 if bmu1 == c1 else c1

        self._finish_conn_update(i_label, bmu1, bmu2)

        self._n_samples += 1
        self._sync_base_cluster_count()

    @_base._add_docs(_base._param_inc_doc)
    def _param_inc(self, sample: np.ndarray, label: int):
        """
        Incremental parameter update for the CONN CVI.
        """

        if self.model_type != "Fuzzy":
            raise ValueError(
                f"model_type={self.model_type!r} supports batch mode only. "
                "Use model_type='Fuzzy' for incremental CONN updates."
            )

        self._update_conn_from_sample(sample, label)

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the CONN CVI.

        Fuzzy batch mode processes samples sequentially because CONN depends
        on online ART category dynamics. KMeans backends first fit fixed
        class-owned prototypes and then accumulate connectivity.
        """

        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if self.normalize_batch:
            data = self._normalize_batch_data(data)

        super()._setup_batch(data)

        # Reset all base and CONN-specific state after setup_batch sets dim.
        self._label_map = _base.LabelMap()
        self._n_samples = 0
        self._n = []
        self._v = np.zeros([0, self._dim])
        self._CP = []
        self._G = np.zeros([0, self._dim])
        self._n_clusters = 0
        self.criterion_value = np.nan

        self._init_conn_state()

        if self.model_type == "Fuzzy":
            for sample, label in zip(data, labels):
                self._update_conn_from_sample(sample, int(label))
        else:
            self._update_conn_from_centroids(data, labels)

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for CONN.

        The CONN value is updated during parameter updates because the update
        requires the label pair touched by the most recent ART transition.
        """

        if self.model_type == "Fuzzy":
            prototype_count = len(self._ensure_artmap().module_a.W)
        else:
            prototype_count = (
                0 if self._cluster_centers is None
                else len(self._cluster_centers)
            )

        if prototype_count <= 1:
            self.criterion_value = np.nan
