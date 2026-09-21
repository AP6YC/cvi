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
from typing import Dict, Literal, Optional, Union
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
    )

    def __init__(
        self,
        rho: float = 0.9,
        alpha: float = 1e-10,
        beta: float = 1.0,
        match_tracking: str = "MT+",
        normalize_batch: bool = True,
        check_incremental_normalized: bool = True,
        model_type: Literal["Fuzzy", "KMeans", "MiniBatchKMeans"] = (
            "MiniBatchKMeans"
        ),
        kmeans_k: Union[int, Dict[int, int]] = 8,
        kmeans_kwargs: Optional[dict] = None,
        *,
        backend: str = "numpy",
    ):
        """
        CONN initialization routine.

        Parameters
        ----------
        rho : float, default=0.9
            FuzzyART vigilance parameter.
        alpha : float, default=1e-10
            FuzzyART choice parameter.
        beta : float, default=1.0
            FuzzyART learning rate.
        match_tracking : str, default="MT+"
            Match-tracking mode passed to SimpleARTMAP.
        normalize_batch : bool, default=True
            If True, batch data are min-max normalized before prototype
            fitting. Incremental data are not normalized online.
        check_incremental_normalized : bool, default=True
            If True, incremental samples are checked to ensure values lie in
            [0, 1].
        model_type : {"Fuzzy", "KMeans", "MiniBatchKMeans"}, default="MiniBatchKMeans"
            Prototype backend. KMeans backends support batch mode only.
        kmeans_k : int or dict[int, int], default=8
            Number of KMeans prototypes per input label. Dictionary values are
            keyed by the original input labels. Counts are capped at the number
            of samples carrying each label.
        kmeans_kwargs : dict, optional
            Keyword arguments forwarded to the selected scikit-learn KMeans
            estimator. ``n_clusters`` must be configured through ``kmeans_k``.
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

        if self.kmeans_kwargs is not None:
            self.kmeans_kwargs = dict(self.kmeans_kwargs)

        self._data_min = None
        self._data_max = None

        self._init_conn_state()

    def _init_conn_state(self):
        """
        Initialize or reset all CONN-specific state.
        """

        from ._conn_art import _CONNFuzzyART, _CONNSimpleARTMAP

        module_a = _CONNFuzzyART(
            rho=self.rho,
            alpha=self.alpha,
            beta=self.beta,
        )
        self._artmap = _CONNSimpleARTMAP(module_a)

        # ART-category-level matrices.
        self._CADJ = _GrowingSquareArray(dtype=float)
        self._CONN = _GrowingSquareArray(dtype=float)

        # Label-level arrays/matrices.
        self._INTRA = _GrowingArray1D(dtype=float)
        self._INTER = _GrowingSquareArray(dtype=float)

        self._intra_conn = 0.0
        self._inter_conn = 0.0

        # Internal label -> set of ART categories assigned to that label.
        self._rev_map = defaultdict(set)

        # Number of samples per internal label.
        self._cluster_cardinality = _GrowingArray1D(dtype=float)

        # Batch centroid-backend state.
        self._kmeans_models = {}
        self._cluster_centers = np.zeros((0, self._dim), dtype=float)
        self._prototype_label_map = {}

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
        """Validate backend selection and KMeans configuration."""

        valid_model_types = {"Fuzzy", "KMeans", "MiniBatchKMeans"}

        if self.model_type not in valid_model_types:
            raise ValueError(
                "model_type must be one of "
                "{'Fuzzy', 'KMeans', 'MiniBatchKMeans'}."
            )

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
            prototype_label_map = self._artmap.map

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
            prototype_count = len(self._artmap.module_a.W)
        else:
            prototype_count = len(self._cluster_centers)

        if prototype_count <= 1:
            self.criterion_value = np.nan
