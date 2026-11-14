"""Pure helpers for scoring candidate updates without mutating a CVI.

An :class:`AddPlan` holds the statistics that adding one sample to each
candidate cluster would produce. Scoring reads it; committing applies one of
its slots. Plans are computed against one exact object state and are only
valid while that state is unchanged.
"""

import numpy as np

_ONE = np.zeros(1, dtype=np.intp)
_NONE = np.zeros(0, dtype=np.intp)
_ONE.flags.writeable = False
_NONE.flags.writeable = False


class AddPlan:
    """New-state statistics for adding one sample to several candidates.

    Candidates are grouped into slots: one per distinct existing internal
    label, plus a single slot (internal label ``-1``) shared by every
    previously unseen label, since each of those creates an identical cluster.
    Index-specific statistics are attached as attributes by the CVI.
    """

    def __init__(self, sample, labels, mapping, fresh):
        self.x = sample
        self.fresh = fresh
        if len(labels) == 1:
            # The committing path plans one label; skip the grouping work.
            label = labels[0]
            self.cand = np.asarray([mapping.get(label, -1)], dtype=np.intp)
            inverse = np.zeros(1, dtype=np.intp)
            self.slot_of = {label: 0}
        else:
            internal = np.fromiter(
                (mapping.get(label, -1) for label in labels),
                dtype=np.intp, count=len(labels),
            )
            self.cand, inverse = np.unique(internal, return_inverse=True)
            self.slot_of = {}
            for label, slot in zip(labels, inverse.tolist()):
                self.slot_of.setdefault(label, slot)
        self._inverse = inverse
        self.n_slots = len(self.cand)
        self.existing = self.cand >= 0
        if self.n_slots == 1:
            self.existing_slots = _ONE if self.existing[0] else _NONE
        else:
            self.existing_slots = np.flatnonzero(self.existing)
        self.existing_labels = self.cand[self.existing_slots]
        self.scores = None

    def scores_for_candidates(self):
        """Scores in the order the candidate labels were supplied."""
        return self.scores[self._inverse]


def dot(left, right):
    """Inner product of two vectors, summed exactly as :func:`row_dot` does."""
    return np.einsum("i,i->", left, right)


def row_dot(left, right):
    """Row-wise inner products of two equally shaped 2-D arrays."""
    return np.einsum("ij,ij->i", left, right)


def min_off_diagonal_excluding(matrix):
    """Return the off-diagonal minimum, and the minimum without each index.

    ``excluded[i]`` is the minimum over pairs that involve neither row nor
    column ``i``. Only the two members of the minimizing pair need a second
    pass, so the cost is a few O(K^2) sweeps.
    """
    k = len(matrix)
    excluded = np.full(k, np.inf)
    if k < 2:
        return np.inf, excluded
    values = matrix.copy()
    np.fill_diagonal(values, np.inf)
    a, b = divmod(int(np.argmin(values)), k)
    total = values[a, b]
    excluded[:] = total
    if k > 2:
        for i in (a, b):
            excluded[i] = np.min(np.delete(np.delete(values, i, 0), i, 1))
    else:
        excluded[[a, b]] = np.inf
    return total, excluded


def row_extremes(matrix, largest):
    """Per-row extreme ignoring the diagonal, with its column and runner-up."""
    fill = -np.inf if largest else np.inf
    values = matrix.copy()
    np.fill_diagonal(values, fill)
    rows = np.arange(len(values))
    column = np.argmax(values, axis=1) if largest else np.argmin(values, axis=1)
    best = values[rows, column]
    values[rows, column] = fill
    second = np.max(values, axis=1) if largest else np.min(values, axis=1)
    return best, column, second


def extremes_without(extremes, excluded_columns):
    """For each excluded column c (one per row of output), each row's extreme
    over the columns other than its diagonal and c."""
    best, column, second = extremes
    return np.where(
        column[None, :] == excluded_columns[:, None],
        second[None, :],
        best[None, :],
    )


def masked_self(rows, plan, fill):
    """Copy candidate rows, replacing each existing candidate's own entry."""
    rows = rows.copy()
    existing = plan.existing
    rows[np.flatnonzero(existing), plan.cand[existing]] = fill
    return rows


def trial_min_off_diagonal(matrix, rows, plan):
    """Off-diagonal minimum after replacing or appending each candidate row.

    ``rows[s]`` holds candidate ``s``'s new entries against the current
    clusters; an existing candidate's own entry is ignored.
    """
    total, excluded = min_off_diagonal_excluding(matrix)
    row_min = np.min(masked_self(rows, plan, np.inf), axis=1, initial=np.inf)
    base = np.full(plan.n_slots, total)
    base[plan.existing] = excluded[plan.cand[plan.existing]]
    return np.minimum(base, row_min)


def top_two(values):
    """Largest value, its index, and the runner-up of a 1-D array."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return -np.inf, -1, -np.inf
    first = int(np.argmax(values))
    rest = np.delete(values, first)
    return values[first], first, (np.max(rest) if len(rest) else -np.inf)


def max_without(values, plan, replacement):
    """Maximum of ``values`` with each existing candidate's entry replaced."""
    best, where, second = top_two(values)
    others = np.full(plan.n_slots, best)
    existing = plan.existing
    others[existing & (plan.cand == where)] = second
    return np.maximum(others, replacement)


def same_arguments(left, right):
    """Exact equality for memo keys holding scalars and arrays."""
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                return False
            if a.shape != b.shape or not np.array_equal(a, b):
                return False
        elif a is None or b is None:
            if a is not b:
                return False
        else:
            try:
                if not bool(a == b):
                    return False
            except (TypeError, ValueError):
                return False
    return True
