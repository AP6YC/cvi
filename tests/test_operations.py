"""Tests for incremental CVI remove and merge operations."""

import copy

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.modules.CONN import CONN


SAMPLES = np.asarray([
    [0.0, 0.0],
    [0.2, 0.1],
    [-0.1, 0.3],
    [1.5, 1.4],
    [1.7, 1.5],
    [1.4, 1.8],
    [3.0, 0.1],
    [3.2, 0.0],
    [2.8, 0.3],
])
LABELS = np.asarray([10, 10, 10, 20, 20, 20, 30, 30, 30])


def build_incrementally(cvi_type, samples=SAMPLES, labels=LABELS):
    """Build one CVI by replaying samples incrementally."""

    local_cvi = cvi_type()
    for sample, label in zip(samples, labels):
        local_cvi.get_cvi(sample, int(label))
    return local_cvi


def assert_equivalent(actual, expected):
    """Compare the common and index-specific sufficient statistics."""

    assert actual._label_map.map == expected._label_map.map
    assert actual._n_samples == expected._n_samples
    assert actual._n_clusters == expected._n_clusters
    np.testing.assert_array_equal(np.asarray(actual._n), np.asarray(expected._n))
    np.testing.assert_allclose(actual._v, expected._v, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        actual.criterion_value,
        expected.criterion_value,
        rtol=1e-7,
        atol=1e-10,
    )

    for attribute in ("_CP", "_G", "_D", "_S", "_SEP", "_sigma"):
        if not hasattr(actual, attribute) or not hasattr(expected, attribute):
            continue

        actual_value = getattr(actual, attribute)
        expected_value = getattr(expected, attribute)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
            continue

        np.testing.assert_allclose(
            np.asarray(actual_value),
            np.asarray(expected_value),
            rtol=1e-7,
            atol=1e-9,
        )


def core_snapshot(local_cvi):
    """Copy state used to prove failed operations are atomic."""

    return {
        "label_map": copy.deepcopy(local_cvi._label_map.map),
        "n_samples": local_cvi._n_samples,
        "n_clusters": local_cvi._n_clusters,
        "n": copy.deepcopy(local_cvi._n),
        "v": local_cvi._v.copy(),
        "criterion_value": local_cvi.criterion_value,
    }


def assert_snapshot(local_cvi, snapshot):
    """Assert that core CVI state agrees with a prior snapshot."""

    assert local_cvi._label_map.map == snapshot["label_map"]
    assert local_cvi._n_samples == snapshot["n_samples"]
    assert local_cvi._n_clusters == snapshot["n_clusters"]
    np.testing.assert_array_equal(np.asarray(local_cvi._n), snapshot["n"])
    np.testing.assert_array_equal(local_cvi._v, snapshot["v"])
    assert local_cvi.criterion_value == snapshot["criterion_value"]


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_remove_matches_incremental_replay(cvi_type):
    """Removing a member must equal replaying all other samples."""

    remove_index = 1
    actual = build_incrementally(cvi_type)
    returned = actual.remove(SAMPLES[remove_index], int(LABELS[remove_index]))

    keep = np.arange(len(SAMPLES)) != remove_index
    expected = build_incrementally(cvi_type, SAMPLES[keep], LABELS[keep])

    assert returned == actual.criterion_value
    assert_equivalent(actual, expected)


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_merge_matches_relabelled_incremental_replay(cvi_type):
    """Merging labels must equal replaying with the source relabelled."""

    actual = build_incrementally(cvi_type)
    returned = actual.merge(target_label=20, source_label=10)

    merged_labels = LABELS.copy()
    merged_labels[merged_labels == 10] = 20
    expected = build_incrementally(cvi_type, SAMPLES, merged_labels)

    assert returned == actual.criterion_value
    assert_equivalent(actual, expected)


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_add_then_remove_restores_state(cvi_type):
    """An add/remove round trip must restore the previous summary."""

    expected = build_incrementally(cvi_type)
    actual = build_incrementally(cvi_type)
    sample = np.asarray([0.05, 0.15])

    actual.get_cvi(sample, 10)
    actual.remove(sample, 10)

    assert_equivalent(actual, expected)


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_merge_to_single_cluster(cvi_type):
    """Merging the final two clusters leaves the CVI undefined at zero."""

    samples = SAMPLES[:6]
    labels = LABELS[:6]
    actual = build_incrementally(cvi_type, samples, labels)
    actual.merge(target_label=20, source_label=10)

    expected = build_incrementally(
        cvi_type,
        samples,
        np.full(labels.shape, 20),
    )

    assert_equivalent(actual, expected)
    assert actual.criterion_value == 0.0


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_singleton_removal_compacts_and_reuses_label(cvi_type):
    """Deleting a singleton removes its mapping and permits label reuse."""

    singleton = np.asarray([[4.0, 4.0]])
    samples = np.vstack((SAMPLES, singleton))
    labels = np.append(LABELS, 99)
    actual = build_incrementally(cvi_type, samples, labels)

    actual.remove(singleton[0], 99)
    assert 99 not in actual._label_map.map
    assert actual._n_clusters == 3
    assert_equivalent(actual, build_incrementally(cvi_type))

    replacement = np.asarray([4.2, 3.9])
    actual.get_cvi(replacement, 99)
    assert actual._label_map.map[99] == 3
    assert actual._n[3] == 1


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_removing_final_sample_resets_object(cvi_type):
    """Removing the final sample returns the object to fresh state."""

    local_cvi = build_incrementally(
        cvi_type,
        samples=np.asarray([[0.25, 0.75]]),
        labels=np.asarray([42]),
    )
    local_cvi.remove(np.asarray([0.25, 0.75]), 42)

    assert local_cvi._n_samples == 0
    assert local_cvi._n_clusters == 0
    assert local_cvi._label_map.map == {}
    assert local_cvi._is_setup is False
    assert local_cvi._mode is None
    assert local_cvi.criterion_value == 0.0

    local_cvi.get_cvi(np.asarray([0.1, 0.2]), 42)
    assert local_cvi._n_samples == 1
    assert local_cvi._label_map.map == {42: 0}


def test_rcip_remove_from_two_sample_cluster():
    """rCIP handles the special covariance transition to a singleton."""

    samples = SAMPLES[:5]
    labels = np.asarray([10, 10, 20, 20, 20])
    actual = build_incrementally(cvi.rCIP, samples, labels)
    actual.remove(samples[0], 10)

    expected = build_incrementally(cvi.rCIP, samples[1:], labels[1:])
    assert_equivalent(actual, expected)


def test_rcip_merge_singletons():
    """Merging rCIP singletons creates the correct sample covariance."""

    samples = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    labels = np.asarray([10, 20, 30])
    actual = build_incrementally(cvi.rCIP, samples, labels)
    actual.merge(target_label=20, source_label=10)

    merged_labels = np.asarray([20, 20, 30])
    expected = build_incrementally(cvi.rCIP, samples, merged_labels)
    assert_equivalent(actual, expected)


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_invalid_operation_arguments_are_atomic(cvi_type):
    """Label and dimension errors must not mutate the object."""

    local_cvi = build_incrementally(cvi_type)
    snapshot = core_snapshot(local_cvi)

    with pytest.raises(ValueError, match="Unknown cluster label"):
        local_cvi.remove(SAMPLES[0], 999)
    assert_snapshot(local_cvi, snapshot)

    with pytest.raises(ValueError, match="Unknown cluster label"):
        local_cvi.merge(10, 999)
    assert_snapshot(local_cvi, snapshot)

    with pytest.raises(ValueError, match="Expected a sample"):
        local_cvi.remove(np.asarray([1.0, 2.0, 3.0]), 10)
    assert_snapshot(local_cvi, snapshot)

    with pytest.raises(ValueError, match="two different"):
        local_cvi.merge(10, 10)
    assert_snapshot(local_cvi, snapshot)


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_operations_require_incremental_state(cvi_type):
    """Fresh and batch-initialized CVIs reject structural operations."""

    fresh = cvi_type()
    with pytest.raises(ValueError, match="initialized incremental"):
        fresh.remove(SAMPLES[0], 10)
    with pytest.raises(ValueError, match="initialized incremental"):
        fresh.merge(10, 20)

    batch = cvi_type()
    contiguous_labels = (LABELS // 10) - 1
    batch.get_cvi(SAMPLES, contiguous_labels)
    with pytest.raises(ValueError, match="batch-initialized"):
        batch.remove(SAMPLES[0], 0)
    with pytest.raises(ValueError, match="batch-initialized"):
        batch.merge(0, 1)


@pytest.mark.parametrize("cvi_type", [cvi.CH, cvi.cSIL, cvi.rCIP])
def test_inconsistent_remove_is_atomic(cvi_type):
    """Statistics-bearing CVIs reject a sample inconsistent with a cluster."""

    local_cvi = build_incrementally(cvi_type)
    snapshot = core_snapshot(local_cvi)

    with pytest.raises(ValueError, match="invalid (cluster compactness|covariance)"):
        local_cvi.remove(np.asarray([100.0, 100.0]), 10)

    assert_snapshot(local_cvi, snapshot)


def test_conn_operations_are_explicitly_unsupported():
    """CONN exposes the common interface but defers reversible state."""

    conn = CONN()

    with pytest.raises(NotImplementedError, match="does not support"):
        conn.remove(np.asarray([0.1, 0.2]), 0)

    with pytest.raises(NotImplementedError, match="does not support"):
        conn.merge(0, 1)
