"""Tests for trial operations, move, lifecycle methods, and accessors."""

import copy
import pickle
import subprocess
import sys
import warnings
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.modules import _base


CVIS_NOT_CONN = [m for m in cvi.MODULES if m is not cvi.CONN]
REPO_ROOT = Path(__file__).resolve().parents[1]


def backend_cases(indices=CVIS_NOT_CONN):
    return [
        pytest.param((index, backend), id=f"{index.__name__}-{backend}")
        for index in indices
        for backend in ("numpy", "numba")
        if backend in index.info.backends
    ]


@pytest.fixture
def cvi_type(request):
    index_type, backend = request.param
    if backend == "numba":
        pytest.importorskip("numba")
    return partial(index_type, backend=backend)


def blobs(seed=0, n=24, d=3):
    """Three separated Gaussian blobs with integer labels 10, 20, 30."""
    rng = np.random.default_rng(seed)
    centers = np.asarray([[0.0] * d, [3.0] * d, [0.0, 4.0] + [1.0] * (d - 2)])
    labels = np.repeat([10, 20, 30], n // 3)
    data = centers[np.repeat([0, 1, 2], n // 3)] + rng.normal(0, 0.4, (n, d))
    order = rng.permutation(n)
    return data[order], labels[order]


def build(cvi_type, data, labels):
    index = cvi_type()
    for sample, label in zip(data, labels):
        index.get_cvi(sample, int(label))
    return index


def assert_same_value(a, b, path="state"):
    """Exact recursive comparison of CVI state values."""
    if isinstance(a, _base.LabelMap) or isinstance(b, _base.LabelMap):
        assert type(a) is type(b), path
        assert a.map == b.map, path
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, path
        np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), path
        for ix, (left, right) in enumerate(zip(a, b)):
            assert_same_value(left, right, f"{path}[{ix}]")
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            assert_same_value(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, float) or isinstance(b, float):
        np.testing.assert_equal(a, b, err_msg=path)
    elif type(a).__module__.startswith("src.cvi.backends"):
        assert type(a) is type(b), path
    elif hasattr(a, "__dict__") and not isinstance(a, type):
        assert type(a) is type(b), path
        assert_same_value(vars(a), vars(b), path)
    else:
        assert a == b, path


def assert_same_state(actual, expected):
    """All state except the private trial memo is identical."""
    left = {k: v for k, v in actual.__dict__.items() if k != "_trial_memo"}
    right = {k: v for k, v in expected.__dict__.items() if k != "_trial_memo"}
    assert left.keys() == right.keys()
    for key in left:
        assert_same_value(left[key], right[key], key)


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12,
                               equal_nan=True)


def candidate_labels(index):
    """Every existing label, one unseen label, and a duplicate."""
    labels = list(index.labels)
    return np.asarray(labels + [99, labels[0]])


# --------------------------------------------------------------------------- #
# Additions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_add_trial_matches_commit_and_leaves_object_unchanged(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-1], labels[:-1])
    sample = data[-1]
    for label in candidate_labels(index):
        before = copy.deepcopy(index)
        score = index.get_cvi(sample, label, commit=False)
        assert isinstance(score, float)
        assert_same_state(index, before)

        committed = copy.deepcopy(before)
        assert_close(score, committed.get_cvi(sample, label))


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_vectorized_add_matches_scalar_trials(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-1], labels[:-1])
    candidates = candidate_labels(index)
    before = copy.deepcopy(index)
    scores = index.get_cvi(data[-1], candidates, commit=False)
    assert_same_state(index, before)
    assert isinstance(scores, np.ndarray) and scores.shape == candidates.shape
    expected = [index.get_cvi(data[-1], label, commit=False)
                for label in candidates]
    assert_close(scores, expected)
    assert index.get_cvi(data[-1], np.asarray([]), commit=False).shape == (0,)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_commit_after_trial_reuses_plan_and_matches_direct_commit(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-1], labels[:-1])
    for label in candidate_labels(index)[:-1]:
        direct = copy.deepcopy(index)
        direct.get_cvi(data[-1], label)

        memoized = copy.deepcopy(index)
        memoized.get_cvi(data[-1], candidate_labels(index), commit=False)
        assert memoized._trial_memo is not None
        memoized.get_cvi(data[-1], label)
        assert memoized._trial_memo is None
        assert_same_state(memoized, direct)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_memoized_commit_does_not_recompute(cvi_type, monkeypatch):
    data, labels = blobs()
    index = build(cvi_type, data[:-1], labels[:-1])
    index.get_cvi(data[-1], candidate_labels(index), commit=False)

    def fail(*args, **kwargs):
        raise AssertionError("commit must reuse the trial plan")

    monkeypatch.setattr(index, "_plan_add", fail)
    index.get_cvi(data[-1], int(labels[0]))


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_stale_trial_is_not_reused(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-2], labels[:-2])
    expected = copy.deepcopy(index)

    index.get_cvi(data[-1], int(labels[-1]), commit=False)
    index.get_cvi(data[-2], int(labels[-2]))
    index.get_cvi(data[-1], int(labels[-1]))

    expected.get_cvi(data[-2], int(labels[-2]))
    expected.get_cvi(data[-1], int(labels[-1]))
    assert_same_state(index, expected)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_trials_on_new_or_single_cluster_objects_return_nan(cvi_type):
    index = cvi_type()
    before = copy.deepcopy(index)
    assert np.isnan(index.get_cvi(np.ones(3), 5, commit=False))
    assert np.all(np.isnan(index.get_cvi(np.ones(3), [5, 6], commit=False)))
    assert_same_state(index, before)

    index.get_cvi(np.ones(3), 5)
    assert np.isnan(index.get_cvi(np.zeros(3), 5, commit=False))
    index.get_cvi(np.zeros(3), 5)
    assert np.isnan(index.criterion_value)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_committing_requires_scalar_label(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-1], labels[:-1])
    before = copy.deepcopy(index)
    with pytest.raises(ValueError, match="scalar label"):
        index.get_cvi(data[-1], np.asarray([10, 20]))
    with pytest.raises(ValueError, match="scalar label"):
        index.move(data[0], int(labels[0]), np.asarray([10, 20]))
    assert_same_state(index, before)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_batch_trial_matches_batch_commit(cvi_type):
    data, labels = blobs()
    index = cvi_type()
    before = copy.deepcopy(index)
    score = index.get_cvi(data, labels, commit=False)
    assert_same_state(index, before)

    expected = cvi_type()
    expected.get_cvi(data, labels)
    assert_close(score, expected.criterion_value)
    assert index.get_cvi(data, labels) == score
    assert_same_state(index, expected)


# --------------------------------------------------------------------------- #
# Move
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_move_matches_remove_then_add(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data, labels)
    sample, old = data[0], int(labels[0])
    for new in candidate_labels(index):
        if new == old:
            continue
        before = copy.deepcopy(index)
        trial = index.move(sample, old, new, commit=False)
        assert_same_state(index, before)

        expected = copy.deepcopy(index)
        expected.remove(sample, old)
        expected_score = expected.get_cvi(sample, new)

        moved = copy.deepcopy(index)
        assert moved.move(sample, old, new) == moved.criterion_value
        assert_same_state(moved, expected)
        assert_close(trial, expected_score)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_vectorized_move_matches_scalar_trials_and_memoized_commit(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data, labels)
    sample, old = data[3], int(labels[3])
    candidates = candidate_labels(index)
    before = copy.deepcopy(index)
    scores = index.move(sample, old, candidates, commit=False)
    assert_same_state(index, before)
    expected = [index.move(sample, old, new, commit=False) for new in candidates]
    assert_close(scores, expected)
    assert_close(scores[candidates == old], index.criterion_value)

    target = next(label for label in candidates if label != old)
    direct = copy.deepcopy(index)
    direct.move(sample, old, target)
    index.move(sample, old, candidates, commit=False)
    index.move(sample, old, target)
    assert_same_state(index, direct)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_move_to_same_label_is_a_no_op(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data, labels)
    before = copy.deepcopy(index)
    label = int(labels[0])
    assert index.move(data[0], label, label, commit=False) == before.criterion_value
    assert index.move(data[0], label, label) == before.criterion_value
    assert_same_state(index, before)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_move_out_of_singleton_is_atomic_and_matches_sequence(cvi_type):
    data, labels = blobs()
    labels = labels.copy()
    labels[0] = 40
    index = build(cvi_type, data, labels)
    assert index.n_clusters == 4

    expected = copy.deepcopy(index)
    expected.remove(data[0], 40)
    expected.get_cvi(data[0], 10)

    trial = index.move(data[0], 40, [10, 40, 50], commit=False)
    assert_close(trial[0], expected.criterion_value)
    assert index.n_clusters == 4
    index.move(data[0], 40, 10)
    assert_same_state(index, expected)
    assert 40 not in index.labels


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_failed_move_leaves_object_unchanged(cvi_type):
    data, labels = blobs()
    labels = labels.copy()
    labels[0] = 40
    index = build(cvi_type, data, labels)
    before = copy.deepcopy(index)
    for bad in (
        lambda: index.move(data[1], 40, 10),          # not the singleton
        lambda: index.move(data[0], 77, 10),          # unknown old label
        lambda: index.move(data[0][:2], 40, 10),      # wrong dimension
    ):
        with pytest.raises(ValueError):
            bad()
        assert_same_state(index, before)


# --------------------------------------------------------------------------- #
# Structural operations
# --------------------------------------------------------------------------- #


def structural_operations(data, labels):
    count = 3
    subset = data[labels == 30][:count]
    centroid = subset.mean(axis=0)
    centered = subset - centroid
    return {
        "remove": lambda index, **kw: index.remove(data[0], int(labels[0]), **kw),
        "merge": lambda index, **kw: index.merge(10, 20, **kw),
        "split": lambda index, **kw: index.split(
            30, 40, count, centroid,
            compactness=float(np.sum(centered ** 2)),
            covariance=centered.T @ centered / (count - 1),
            **kw,
        ),
    }


@pytest.mark.parametrize("operation", ["remove", "merge", "split"])
@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_structural_trial_matches_commit(cvi_type, operation):
    data, labels = blobs()
    run = structural_operations(data, labels)[operation]
    index = build(cvi_type, data, labels)
    before = copy.deepcopy(index)
    trial = run(index, commit=False)
    assert_same_state(index, before)

    direct = copy.deepcopy(before)
    assert_close(trial, run(direct))
    np.testing.assert_equal(trial, run(index))
    assert_same_state(index, direct)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_trial_operations_require_initialization(cvi_type):
    index = cvi_type()
    with pytest.raises(ValueError, match="initialized"):
        index.remove(np.zeros(2), 1, commit=False)
    with pytest.raises(ValueError, match="initialized"):
        index.move(np.zeros(2), 1, 2, commit=False)


# --------------------------------------------------------------------------- #
# Random sequences against batch
# --------------------------------------------------------------------------- #


def batch_score(cvi_type, members):
    data = np.asarray([sample for sample, _ in members])
    labels = np.asarray([label for _, label in members])
    if len(np.unique(labels)) < 2:
        return np.nan
    index = cvi_type()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return index.get_cvi(data, labels)


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_random_operation_sequences_match_batch(cvi_type, seed):
    rng = np.random.default_rng(seed)
    data, true_labels = blobs(seed=seed, n=30)
    index = cvi_type()
    members = []          # (sample, label) currently counted
    pending = list(range(len(data)))

    def check(score):
        assert index.n_samples == len(members)
        assert set(index.labels) == {label for _, label in members}
        np.testing.assert_allclose(
            score, batch_score(cvi_type, members),
            rtol=1e-7, atol=1e-9, equal_nan=True,
        )

    for _ in range(80):
        current = sorted({label for _, label in members})
        choice = rng.choice(["add", "move", "remove", "merge"],
                            p=[0.45, 0.35, 0.1, 0.1])
        if choice == "add" or len(members) < 4:
            if not pending:
                continue
            row = pending.pop(int(rng.integers(len(pending))))
            label = int(rng.choice(current + [int(true_labels[row]), 50 + _]))
            candidates = np.asarray(current + [label])
            scores = index.get_cvi(data[row], candidates, commit=False)
            score = index.get_cvi(data[row], label)
            assert_close(scores[-1], score)
            members.append((data[row], label))
        elif choice == "move":
            position = int(rng.integers(len(members)))
            sample, old = members[position]
            new = int(rng.choice(current + [60 + _]))
            candidates = np.asarray(current + [new])
            scores = index.move(sample, old, candidates, commit=False)
            score = index.move(sample, old, new)
            assert_close(scores[-1], score)
            members[position] = (sample, new)
        elif choice == "remove":
            position = int(rng.integers(len(members)))
            sample, old = members.pop(position)
            trial = index.remove(sample, old, commit=False)
            score = index.remove(sample, old)
            np.testing.assert_equal(trial, score)
            pending.append(int(np.flatnonzero((data == sample).all(axis=1))[0]))
        else:
            if len(current) < 2:
                continue
            target, source = (int(v) for v in rng.choice(current, 2, replace=False))
            trial = index.merge(target, source, commit=False)
            score = index.merge(target, source)
            np.testing.assert_equal(trial, score)
            members = [(s, target if l == source else l) for s, l in members]
        if members:
            check(index.criterion_value)


# --------------------------------------------------------------------------- #
# Lifecycle, accessors, capabilities
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_reset_and_clone(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data, labels)
    backend = index._backend

    fresh = index.clone()
    assert type(fresh) is type(index) and fresh.backend == index.backend
    assert fresh.n_samples == 0 and fresh.n_clusters == 0
    assert index.n_samples == len(data)

    assert index.reset() is index
    assert index._backend is backend
    assert index.n_samples == 0 and index.labels == ()
    assert np.isnan(index.criterion_value)

    expected = cvi_type()
    assert_close(index.get_cvi(data, labels), expected.get_cvi(data, labels))


def test_clone_and_reset_keep_constructor_configuration():
    index = cvi.CONN(model_type="KMeans", kmeans_k=2, normalize_batch=False)
    data, labels = blobs()
    index.get_cvi(data, labels)
    for other in (index.clone(), index.reset()):
        assert other.model_type == "KMeans"
        assert other.kmeans_k == 2
        assert other.normalize_batch is False
        assert other.n_samples == 0


def test_accessors():
    index = cvi.CH()
    assert (index.n_clusters, index.n_samples, index.labels) == (0, 0, ())
    for sample, label in zip(np.eye(3), [7, 3, 7]):
        index.get_cvi(sample, label)
    assert index.n_clusters == 2
    assert index.n_samples == 3
    assert index.labels == (7, 3)


@pytest.mark.parametrize(
    "index_type, new, old, tol, expected",
    [
        (cvi.CH, 2.0, 1.0, 0.0, True),
        (cvi.CH, 1.0, 1.0, 0.0, True),
        (cvi.CH, 0.9, 1.0, 0.0, False),
        (cvi.CH, 0.9, 1.0, 0.2, True),
        (cvi.DB, 0.5, 1.0, 0.0, True),
        (cvi.DB, 1.0, 1.0, 0.0, True),
        (cvi.DB, 1.1, 1.0, 0.0, False),
        (cvi.DB, 1.1, 1.0, 0.2, True),
        (cvi.CH, np.nan, 1.0, 0.0, True),
        (cvi.DB, 1.0, np.nan, 0.0, True),
    ],
)
def test_is_better(index_type, new, old, tol, expected):
    assert index_type.is_better(new, old, tol) is expected
    assert index_type().is_better(new, old, tol=tol) is expected


def test_conn_reports_no_trial_or_move_support():
    index = cvi.CONN(model_type="KMeans", kmeans_k=2)
    assert not index.capabilities.trial and not index.capabilities.move
    data, labels = blobs()
    index.get_cvi(data, labels)
    for call in (
        lambda: index.get_cvi(data[0], 10, commit=False),
        lambda: index.merge(10, 20, commit=False),
        lambda: index.split(10, 40, [0], commit=False),
        lambda: index.remove(data[0], 10, commit=False),
        lambda: index.move(data[0], 10, 20),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_jax_reports_no_trial_or_move_support(monkeypatch):
    monkeypatch.setattr(_base, "get_backend",
                        lambda backend: SimpleNamespace(name=backend))
    index = cvi.CH(backend="jax", capacity=4)
    assert not index.capabilities.trial and not index.capabilities.move
    with pytest.raises(NotImplementedError, match="commit=False"):
        index.get_cvi(np.zeros(2), 1, commit=False)
    with pytest.raises(NotImplementedError):
        index.move(np.zeros(2), 1, 2)


@pytest.mark.parametrize("cvi_type", backend_cases(), indirect=True)
def test_update_batch_discards_trial(cvi_type):
    data, labels = blobs()
    index = build(cvi_type, data[:-3], labels[:-3])
    index.get_cvi(data[-1], int(labels[-1]), commit=False)
    index.update_batch(data[-3:-1], labels[-3:-1])
    assert index._trial_memo is None


# --------------------------------------------------------------------------- #
# Hygiene
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index_type", cvi.MODULES)
def test_pickle_round_trip(index_type):
    data, labels = blobs()
    if index_type is cvi.CONN:
        index = cvi.CONN(model_type="KMeans", kmeans_k=2)
        index.get_cvi(data, labels)
    else:
        index = build(index_type, data[:-1], labels[:-1])
        index.get_cvi(data[-1], [10, 99], commit=False)

    restored = pickle.loads(pickle.dumps(index))
    assert restored._trial_memo is None
    assert_same_state(restored, index)
    assert restored.clone().n_samples == 0
    if index_type is not cvi.CONN:
        assert_close(restored.get_cvi(data[-1], 10), index.get_cvi(data[-1], 10))


def test_import_does_not_import_artlib():
    code = (
        "import sys; import src.cvi as cvi; "
        "[m() for m in cvi.MODULES if m is not cvi.CONN]; cvi.CONN(); "
        "assert not any(name.split('.')[0] == 'artlib' for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


# --------------------------------------------------------------------------- #
# Edge cases and helpers
# --------------------------------------------------------------------------- #


def test_clone_supports_subclasses_and_older_pickles():
    class Custom(cvi.CH):
        pass

    index = Custom(backend="numba") if "numba" in sys.modules else Custom()
    clone = index.clone()
    assert type(clone) is Custom and clone.backend == index.backend

    legacy = cvi.CH()
    del legacy.__dict__["_init_params"]
    assert legacy.clone().backend == "numpy" and legacy.reset().n_samples == 0


def test_invalid_trial_inputs_raise_without_changes():
    data, labels = blobs()
    index = build(cvi.CH, data, labels)
    before = copy.deepcopy(index)
    with pytest.raises(ValueError, match="one-dimensional array"):
        index.get_cvi(data[0], np.asarray([[10, 20]]), commit=False)
    with pytest.raises(ValueError, match="features"):
        index.get_cvi(data[0][:2], 10, commit=False)
    with pytest.raises(ValueError, match="ndim=4"):
        index.get_cvi(data[None, None], 10, commit=False)
    with pytest.raises(ValueError, match="Repeated batch"):
        index.get_cvi(data, labels, commit=False)
    assert_same_state(index, before)


def test_undefined_batch_trial_warns_like_a_batch_commit():
    data = np.ones((4, 2))
    labels = np.asarray([0, 0, 1, 1])
    index = cvi.CH()
    with pytest.warns(RuntimeWarning, match="undefined"):
        assert np.isnan(index.get_cvi(data, labels, commit=False))
    with pytest.warns(RuntimeWarning, match="undefined"):
        assert np.isnan(index.get_cvi(data, labels))


def test_stale_plan_is_rejected():
    data, labels = blobs()
    index = build(cvi.CH, data[:-1], labels[:-1])
    plan = index._plan_add(data[-1], [99], score=False)
    index.get_cvi(data[-1], 99)
    with pytest.raises(RuntimeError, match="no longer matches"):
        index._apply_add(plan, 0, 99)


def test_trial_helpers():
    from src.cvi.modules import _trial

    a = np.arange(3.0)
    assert _trial.same_arguments((a, 1, None), (a.copy(), 1, None))
    assert not _trial.same_arguments((a,), (a, 1))
    assert not _trial.same_arguments((a,), (list(a),))
    assert not _trial.same_arguments((a,), (a[:2],))
    assert not _trial.same_arguments((None,), (0,))
    assert not _trial.same_arguments((1,), (2,))
    assert not _trial.same_arguments((a.tolist(),), ([0.0, 1.0],))
    assert not _trial.same_arguments(([a, a],), ([a.copy(), a],))  # ambiguous
    assert _trial.top_two([]) == (-np.inf, -1, -np.inf)
    assert _trial.top_two([3.0]) == (3.0, 0, -np.inf)
    assert _trial.min_off_diagonal_excluding(np.zeros((1, 1)))[0] == np.inf
