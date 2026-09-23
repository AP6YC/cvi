"""ART adapters loaded only when constructing a CONN index."""

from typing import Optional

import numpy as np
from artlib import FuzzyART, SimpleARTMAP


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
