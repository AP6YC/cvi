"""Pure helpers for scoring candidate updates without mutating a CVI.

An :class:`AddPlan` holds the statistics that adding one sample to each
candidate cluster would produce. Scoring reads it; committing applies one of
its slots. Plans are computed against one exact object state and are only
valid while that state is unchanged.
"""

import numpy as np

if hasattr(np, "vecdot"):
    _row_inner = np.vecdot
else:  # pragma: no cover - NumPy < 2.0
    def _row_inner(left, right):
        return np.asarray([np.inner(a, b) for a, b in zip(left, right)])

_ONE = np.zeros(1, dtype=np.intp)
_ONE.flags.writeable = False


class AddPlan:
    """New-state statistics for adding one sample to several candidates.

    Candidates are grouped into slots: one per distinct existing internal
    label, plus a single slot (internal label ``-1``) shared by every
    previously unseen label, since each of those creates an identical cluster.
    Index-specific statistics are attached as attributes by the CVI.

    A one-label plan (every plain commit) keeps its per-slot values in
    one-element lists, which are much cheaper to build and index than 1-row
    arrays; :meth:`vectorize` turns them into arrays before scoring. Lists
    are therefore reserved for per-slot values.
    """

    def __init__(self, sample, labels, mapping, fresh):
        self.x = sample
        self.fresh = fresh
        # Index-specific rows against the current clusters; set only when the
        # plan is scored, and otherwise computed for the applied slot alone.
        self.rows = None
        self.scores = None
        if len(labels) == 1:
            label = labels[0]
            self.cand = [mapping.get(label, -1)]
            self.slot_of = {label: 0}
            self.n_slots = 1
            self._inverse = _ONE
            return
        internal = np.fromiter(
            (mapping.get(label, -1) for label in labels),
            dtype=np.intp, count=len(labels),
        )
        self.cand, self._inverse = np.unique(internal, return_inverse=True)
        self.slot_of = {}
        for label, slot in zip(labels, self._inverse.tolist()):
            self.slot_of.setdefault(label, slot)
        self.n_slots = len(self.cand)
        self._index_slots()

    def _index_slots(self):
        self.existing = self.cand >= 0
        self.existing_slots = np.flatnonzero(self.existing)
        self.existing_labels = self.cand[self.existing_slots]

    def vectorize(self):
        """Convert one-element per-slot lists into arrays for scoring."""
        for name, value in list(vars(self).items()):
            if isinstance(value, list):
                setattr(self, name, np.asarray(value))
        if not hasattr(self, "existing"):
            self._index_slots()

    def scores_for_candidates(self):
        """Scores in the order the candidate labels were supplied."""
        return self.scores[self._inverse]


def inner_products(vectors):
    """``np.inner`` for vectors, or its row-wise form for 2-D arrays.

    ``np.vecdot`` runs the same inner-product loop as ``np.inner`` on every
    row, so a one-slot update (as streamed since 0.7) and the same slot of a
    many-slot plan agree bit for bit.
    """
    return np.inner if vectors.ndim == 1 else _row_inner


def centroid_update(centroid, sample, count, shift_form=False):
    """Centroid(s) after adding ``sample``; ``count`` is the new count.

    ``centroid`` is one vector with a scalar count, or one row per slot with
    a column of counts.
    """
    if shift_form:
        return centroid + (sample - centroid) / count
    return (1 - 1 / count) * centroid + (1 / count) * sample


def compactness_update(compactness, moment, centroid, centroid_new, sample,
                       count):
    """Compactness and first-moment term after adding ``sample``.

    ``count`` is the count before the addition: a scalar for one vector, or
    one entry per row.
    """
    delta_v = centroid - centroid_new
    diff_x_v = sample - centroid_new
    inner = inner_products(diff_x_v)
    column = count if diff_x_v.ndim == 1 else count[:, None]
    compactness_new = (
        compactness
        + inner(diff_x_v, diff_x_v)
        + count * inner(delta_v, delta_v)
        + 2 * inner(delta_v, moment)
    )
    return compactness_new, moment + diff_x_v + column * delta_v


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
