"""
Connectivity-based CONN Cluster Validity Index.

This implementation follows the incremental CONN-style validity index for
ART/ARTMAP partitions. Unlike distance-only CVIs, CONN depends on the first
and second best matching ART categories associated with each sample.

Notes
-----
Incremental mode assumes samples are already normalized to the ART input
domain, typically [0, 1]. Batch mode can optionally normalize the full dataset
before processing.

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
from typing import Optional
import numbers

# Third-party imports
import numpy as np

# ART imports
from artlib import FuzzyART, SimpleARTMAP
from artlib.common.utils import complement_code

# Local imports
from . import _base


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


class _CONNFuzzyART(FuzzyART):
    """
    FuzzyART extension that exposes the first and second best matching
    categories for CONN updates.
    """

    def step_pred_first_and_second(self, sample: np.ndarray):
        """
        Return the first and second best matching ART categories.

        Parameters
        ----------
        sample : np.ndarray
            Complement-coded sample.

        Returns
        -------
        tuple[int, int]
            First and second category indices.

        Raises
        ------
        RuntimeError
            If fewer than two ART categories exist.
        """

        if len(self.W) < 2:
            raise RuntimeError(
                "CONN requires at least two ART categories. "
                "The second ART category should be forced during the "
                "second-sample initialization step."
            )

        choices = [
            self.category_choice(sample, w, params=self.params)[0]
            for w in self.W
        ]

        choices = np.asarray(choices, dtype=float)

        first = int(np.argmax(choices))
        choices[first] = -np.inf
        second = int(np.argmax(choices))

        return first, second


class _CONNSimpleARTMAP(SimpleARTMAP):
    """
    SimpleARTMAP extension with CONN-specific match reset behavior.
    """

    def match_reset_func(
        self,
        i: np.ndarray,
        w: np.ndarray,
        cluster_a,
        params: dict,
        extra: dict,
        cache: Optional[dict] = None,
    ) -> bool:
        """
        CONN-specific match reset.
        """

        cluster_b = extra["cluster_b"]

        b_samples = sum(
            self.module_a.weight_sample_counter_[a]
            for a, b in self.map.items()
            if b == cluster_b
        )

        if b_samples == 1:
            return False

        if cluster_a in self.map and self.map[cluster_a] != cluster_b:
            return False

        return True


class CONN(_base.CVI):
    """
    Incremental CONN Cluster Validity Index.

    This CVI is ART-dependent. It internally maintains a FuzzyART/SimpleARTMAP
    model in order to obtain the first and second best matching ART categories
    needed by the CONN update.

    References
    ----------
    1. E. Merényi, "A new cluster validity index for prototype based clustering algorithms based on inter-and intra-cluster density," 2007 International Joint Conference on Neural Networks, 2007.
    2. K. Tasdemir and E. Merényi, "A validity index for prototype-based clustering of data sets with complex cluster structures," IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 41, no. 4, pp. 1039-1053, 2011.
    3. L. E. Brito da Silva, N. M. Melton, and D. C. Wunsch II, "Incremental cluster validity indices for online learning of hard partitions: Extensions and comparative study," IEEE Access, vol. 8, pp. 22025-22047, 2020.
    """

    def __init__(
        self,
        rho: float = 0.9,
        alpha: float = 1e-10,
        beta: float = 1.0,
        match_tracking: str = "MT+",
        normalize_batch: bool = True,
        check_incremental_normalized: bool = True,
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
            If True, batch data are min-max normalized before ART processing.
            Incremental data are not normalized online.
        check_incremental_normalized : bool, default=True
            If True, incremental samples are checked to ensure values lie in
            [0, 1].
        """

        super().__init__()

        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.match_tracking = match_tracking
        self.normalize_batch = normalize_batch
        self.check_incremental_normalized = check_incremental_normalized

        self._data_min = None
        self._data_max = None

        self._init_conn_state()

    def _init_conn_state(self):
        """
        Initialize or reset all CONN-specific state.
        """

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

    def _finish_conn_update(self, i_label: int, bmu1: int, bmu2: int):
        """
        Finish the CONN bookkeeping once BMU1 and BMU2 are known.
        """

        self._rev_map[i_label].add(bmu1)
        self._cluster_cardinality.increment(i_label, 1)

        self._CADJ.increment(bmu1, bmu2, 1)

        # CONN is the symmetrized co-adjacency.
        conn_value = self._CADJ[bmu1, bmu2] + self._CADJ[bmu2, bmu1]
        self._CONN[bmu1, bmu2] = conn_value
        self._CONN[bmu2, bmu1] = conn_value

        if bmu1 not in self._artmap.map:
            raise RuntimeError("BMU1 is missing from the ARTMAP label map.")

        if int(self._artmap.map[bmu1]) != i_label:
            raise RuntimeError(
                "Internal ARTMAP mapping disagrees with the provided label."
            )

        if bmu2 in self._artmap.map:
            y2 = int(self._artmap.map[bmu2])
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

            self.criterion_value = 0.0
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

        self._update_conn_from_sample(sample, label)

    @_base._add_docs(_base._param_batch_doc)
    def _param_batch(self, data: np.ndarray, labels: np.ndarray):
        """
        Batch parameter update for the CONN CVI.

        Batch mode processes the samples sequentially because CONN depends on
        the online ART category dynamics.
        """

        data = np.asarray(data, dtype=float)
        labels = np.asarray(labels)

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
        self.criterion_value = 0.0

        self._init_conn_state()

        for sample, label in zip(data, labels):
            self._update_conn_from_sample(sample, int(label))

    @_base._add_docs(_base._evaluate_doc)
    def _evaluate(self):
        """
        Criterion value evaluation method for CONN.

        The CONN value is updated during parameter updates because the update
        requires the label pair touched by the most recent ART transition.
        """

        if self._n_clusters <= 0:
            self.criterion_value = 0.0