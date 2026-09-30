"""Focused tests for the CONN validity index."""

import numpy as np
import pytest

import src.cvi as cvi


KMEANS_KWARGS = {"random_state": 0, "n_init": 1}


def _separated_data():
    data = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )
    labels = np.asarray([10, 10, 20, 20])
    return data, labels


def test_only_selected_prototype_defaults_are_resolved():
    fuzzy = cvi.CONN(model_type="Fuzzy")
    assert fuzzy.rho == 0.9
    assert fuzzy.alpha == 1e-10
    assert fuzzy.beta == 1.0
    assert fuzzy.match_tracking == "MT+"
    assert fuzzy.kmeans_k is None
    assert fuzzy.kmeans_kwargs is None

    kmeans = cvi.CONN(model_type="KMeans")
    assert kmeans.rho is None
    assert kmeans.alpha is None
    assert kmeans.beta is None
    assert kmeans.match_tracking is None
    assert kmeans.kmeans_k == 8
    assert kmeans.kmeans_kwargs is None


def test_kmeans_does_not_initialize_art_prototypes():
    data, labels = _separated_data()
    conn = cvi.CONN(
        model_type="KMeans",
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    assert conn._artmap is None
    assert conn._kmeans_models is None

    conn.get_cvi(data, labels)

    assert conn._artmap is None
    assert len(conn._kmeans_models) == 2


def test_fuzzy_initializes_only_art_prototypes_on_first_use():
    conn = cvi.CONN(model_type="Fuzzy", normalize_batch=False)

    assert conn._artmap is None
    assert conn._kmeans_models is None

    conn.get_cvi(np.asarray([0.0, 0.0]), 0)

    assert conn._artmap is not None
    assert conn._kmeans_models is None


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_backends_compute_expected_connectivity(model_type):
    data, labels = _separated_data()
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    value = conn.get_cvi(data, labels)

    expected_cadj = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert value == pytest.approx(1.0)
    np.testing.assert_array_equal(conn._CADJ.asarray(), expected_cadj)
    np.testing.assert_array_equal(
        conn._CONN.asarray(),
        expected_cadj + expected_cadj.T,
    )
    np.testing.assert_array_equal(conn._INTRA.asarray(), [1.0, 1.0])
    np.testing.assert_array_equal(conn._INTER.asarray(), np.zeros((2, 2)))


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_backends_detect_cross_label_connectivity(model_type):
    data = np.asarray(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [0.1, 0.0],
            [10.1, 10.0],
        ]
    )
    labels = np.asarray([10, 10, 20, 20])
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    assert conn.get_cvi(data, labels) == pytest.approx(0.0)
    np.testing.assert_array_equal(conn._INTRA.asarray(), [0.0, 0.0])
    np.testing.assert_array_equal(
        conn._INTER.asarray(),
        [[0.0, 1.0], [1.0, 0.0]],
    )


def test_kmeans_counts_are_per_external_label_and_capped_by_support():
    data = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.2, 10.0],
        ]
    )
    labels = np.asarray([10, 10, 20, 20, 20])
    conn = cvi.CONN(
        model_type="KMeans",
        kmeans_k={10: 8, 20: 1},
        kmeans_kwargs=KMEANS_KWARGS,
    )

    conn.get_cvi(data, labels)

    assert conn._kmeans_models[10].n_clusters == 2
    assert conn._kmeans_models[20].n_clusters == 1
    assert len(conn._rev_map[conn._label_map.map[10]]) == 2
    assert len(conn._rev_map[conn._label_map.map[20]]) == 1
    assert conn._cluster_centers.shape == (3, 2)


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_kmeans_kwargs_are_forwarded(model_type):
    data, labels = _separated_data()
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=2,
        kmeans_kwargs={"random_state": 17, "n_init": 1},
    )

    conn.get_cvi(data, labels)

    for model in conn._kmeans_models.values():
        assert model.random_state == 17
        assert model.n_init == 1


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_batch_normalization_is_affine_scale_invariant(model_type):
    data, labels = _separated_data()
    scaled_data = data * np.asarray([3.0, 7.0]) + np.asarray([-4.0, 20.0])
    kwargs = {
        "model_type": model_type,
        "kmeans_k": 2,
        "kmeans_kwargs": KMEANS_KWARGS,
    }

    conn = cvi.CONN(**kwargs)
    scaled_conn = cvi.CONN(**kwargs)
    value = conn.get_cvi(data, labels)
    scaled_value = scaled_conn.get_cvi(scaled_data, labels)

    assert scaled_value == pytest.approx(value)
    np.testing.assert_array_equal(
        scaled_conn._CADJ.asarray(),
        conn._CADJ.asarray(),
    )


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_backends_reject_incremental_updates_without_mutation(
    model_type,
):
    conn = cvi.CONN(model_type=model_type)

    with pytest.raises(ValueError, match="supports batch mode only"):
        conn.get_cvi(np.asarray([0.0, 0.0]), 0)

    assert conn._is_setup is False
    assert conn._n_samples == 0
    assert conn._label_map.map == {}


def test_fuzzy_backend_retains_batch_and_incremental_behavior():
    data = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    labels = np.asarray([0, 0, 1, 1])
    incremental = cvi.CONN(model_type="Fuzzy", normalize_batch=False)
    batch = cvi.CONN(model_type="Fuzzy", normalize_batch=False)

    for sample, label in zip(data, labels):
        incremental.get_cvi(sample, label)

    batch_value = batch.get_cvi(data, labels)

    assert incremental.criterion_value == pytest.approx(0.125)
    assert batch_value == pytest.approx(incremental.criterion_value)
    np.testing.assert_array_equal(
        batch._CADJ.asarray(),
        incremental._CADJ.asarray(),
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"model_type": "unknown"}, "model_type must be one of"),
        ({"kmeans_k": 0}, "must be a positive integer"),
        ({"kmeans_k": False}, "must be a positive integer"),
        ({"kmeans_k": {0: 0}}, "must be a positive integer"),
        ({"kmeans_k": {"0": 1}}, "dictionary keys must be integers"),
        ({"kmeans_kwargs": []}, "must be a dictionary or None"),
        (
            {"kmeans_kwargs": {"n_clusters": 2}},
            "Configure n_clusters through kmeans_k",
        ),
    ],
)
def test_invalid_backend_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cvi.CONN(**kwargs)


def test_kmeans_dictionary_must_cover_observed_labels():
    data, labels = _separated_data()
    conn = cvi.CONN(
        model_type="KMeans",
        kmeans_k={10: 2},
        kmeans_kwargs=KMEANS_KWARGS,
    )

    with pytest.raises(ValueError, match="missing values for labels.*20"):
        conn.get_cvi(data, labels)


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_duplicate_centers_retain_full_prototype_matrices(model_type):
    data = np.zeros((6, 2))
    labels = np.asarray([10, 10, 10, 20, 20, 20])
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=3,
        kmeans_kwargs=KMEANS_KWARGS,
    )

    value = conn.get_cvi(data, labels)

    assert np.isfinite(value)
    assert conn._cluster_centers.shape == (6, 2)
    assert conn._CADJ.asarray().shape == (6, 6)
    assert conn._CONN.asarray().shape == (6, 6)
    assert len(conn._prototype_label_map) == 6


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_split_and_merge_reassign_fixed_prototypes(model_type):
    data, labels = _separated_data()
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )
    assert conn.get_cvi(data, labels) == pytest.approx(1.0)
    assert conn.get_prototype_ids(10) == (0, 1)
    np.testing.assert_array_equal(conn._prototype_cardinality, [1, 1, 1, 1])
    centers_before = conn._cluster_centers.copy()
    cadj_before = conn._CADJ.asarray()
    conn_before = conn._CONN.asarray()

    assert conn.split(10, 30, [0]) == pytest.approx(1 / 9)
    assert conn.get_prototype_ids(10) == (1,)
    assert conn.get_prototype_ids(30) == (0,)
    np.testing.assert_array_equal(
        conn._cluster_cardinality.asarray(), [1, 2, 1]
    )
    np.testing.assert_array_equal(conn._INTRA.asarray(), [0, 1, 0])
    np.testing.assert_array_equal(
        conn._INTER.asarray(), [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    )
    np.testing.assert_array_equal(conn._cluster_centers, centers_before)
    np.testing.assert_array_equal(conn._CADJ.asarray(), cadj_before)
    np.testing.assert_array_equal(conn._CONN.asarray(), conn_before)

    assert conn.merge(10, 30) == pytest.approx(1.0)
    assert conn._label_map.map == {10: 0, 20: 1}
    assert conn.get_prototype_ids(10) == (0, 1)
    np.testing.assert_array_equal(conn._cluster_cardinality.asarray(), [2, 2])


@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_merge_compacts_internal_labels(model_type):
    data = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0],
         [5.1, 5.0], [10.0, 10.0], [10.1, 10.0]]
    )
    labels = np.asarray([10, 10, 20, 20, 30, 30])
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=2,
        kmeans_kwargs=KMEANS_KWARGS,
    )
    conn.get_cvi(data, labels)

    assert conn.merge(target_label=30, source_label=10) == pytest.approx(1.0)
    assert conn._label_map.map == {20: 0, 30: 1}
    assert conn.get_prototype_ids(20) == (2, 3)
    assert conn.get_prototype_ids(30) == (0, 1, 4, 5)
    np.testing.assert_array_equal(conn._cluster_cardinality.asarray(), [2, 4])


@pytest.mark.filterwarnings("ignore:Number of distinct clusters")
@pytest.mark.parametrize("model_type", ["KMeans", "MiniBatchKMeans"])
def test_centroid_split_requires_sample_support_on_both_sides(model_type):
    data = np.zeros((6, 2))
    labels = np.asarray([10, 10, 10, 20, 20, 20])
    conn = cvi.CONN(
        model_type=model_type,
        kmeans_k=3,
        kmeans_kwargs=KMEANS_KWARGS,
    )
    conn.get_cvi(data, labels)
    mapping_before = dict(conn._prototype_label_map)
    score_before = conn.criterion_value
    np.testing.assert_array_equal(
        conn._prototype_cardinality, [3, 0, 0, 3, 0, 0]
    )

    for ids in ([1], [0]):
        with pytest.raises(ValueError, match="assigned samples"):
            conn.split(10, 30, ids)
        assert conn._prototype_label_map == mapping_before
        assert conn._label_map.map == {10: 0, 20: 1}
        assert conn.criterion_value == score_before


def test_conn_is_public_and_defaults_to_minibatch_kmeans():
    assert cvi.modules.CONN is cvi.CONN
    assert cvi.CONN in cvi.MODULES
    assert cvi.CONN().model_type == "MiniBatchKMeans"
    assert cvi.CONN.info.name_short == "CONN"
    assert cvi.CONN.info.index_min == 0.0
    assert cvi.CONN.info.index_max == 1.0
    assert cvi.CONN.info.optimality == "max"
    assert cvi.CONN.info.merge
    assert cvi.CONN.info.split
    assert not cvi.CONN.info.remove


def _fuzzy_operation_fixture():
    data = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
         [0.05, 0.95], [0.95, 0.05]]
    )
    labels = np.asarray([10, 10, 20, 20, 10, 20])
    conn = cvi.CONN(model_type="Fuzzy", normalize_batch=False, rho=0.9)
    conn.get_cvi(data, labels)
    return conn


def test_fuzzy_split_moves_prototype_without_changing_connectivity():
    conn = _fuzzy_operation_fixture()
    assert conn.get_prototype_ids(10) == (0, 1)
    cadj_before = conn._CADJ.asarray()
    conn_before = conn._CONN.asarray()
    weights_before = [weight.copy() for weight in conn._artmap.module_a.W]
    a_labels_before = conn._artmap.module_a.labels_.copy()

    value = conn.split(retained_label=10, new_label=30, prototype_ids=[1])

    assert value == pytest.approx(0.0)
    assert conn._label_map.map == {10: 0, 20: 1, 30: 2}
    assert conn.get_prototype_ids(10) == (0,)
    assert conn.get_prototype_ids(30) == (1,)
    assert {key: set(value) for key, value in conn._rev_map.items()} == {
        0: {0}, 1: {2, 3}, 2: {1}
    }
    np.testing.assert_array_equal(conn._cluster_cardinality.asarray(), [1, 3, 2])
    np.testing.assert_array_equal(conn._artmap.labels_, [0, 2, 1, 1, 2, 1])
    np.testing.assert_array_equal(conn._artmap.classes_, [0, 1, 2])
    np.testing.assert_array_equal(conn._INTRA.asarray(), [0, 0, 0])
    np.testing.assert_array_equal(
        conn._INTER.asarray(), [[0, 0, 0], [1, 0, 1], [1, 0, 0]]
    )
    np.testing.assert_array_equal(conn._CADJ.asarray(), cadj_before)
    np.testing.assert_array_equal(conn._CONN.asarray(), conn_before)
    np.testing.assert_array_equal(conn._artmap.module_a.labels_, a_labels_before)
    for before, after in zip(weights_before, conn._artmap.module_a.W):
        np.testing.assert_array_equal(after, before)

    conn.get_cvi(np.asarray([0.04, 0.96]), 30)
    assert conn._artmap.map[int(conn._artmap.module_a.labels_[-1])] == 2
    assert conn._cluster_cardinality.asarray()[2] == 3


def test_fuzzy_split_then_merge_restores_score_and_assignments():
    conn = _fuzzy_operation_fixture()
    score_before = conn.criterion_value
    intra_before = conn._INTRA.asarray()
    inter_before = conn._INTER.asarray()

    conn.split(10, 30, [1])
    value = conn.merge(target_label=10, source_label=30)

    assert value == pytest.approx(score_before)
    assert conn._label_map.map == {10: 0, 20: 1}
    np.testing.assert_array_equal(conn._INTRA.asarray(), intra_before)
    np.testing.assert_array_equal(conn._INTER.asarray(), inter_before)
    np.testing.assert_array_equal(conn._artmap.labels_, [0, 0, 1, 1, 0, 1])


def test_fuzzy_split_can_move_multiple_prototypes():
    data = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5],
         [1.0, 0.0], [1.0, 1.0], [0.95, 0.05]]
    )
    labels = np.asarray([10, 10, 10, 10, 20, 20])
    conn = cvi.CONN(model_type="Fuzzy", normalize_batch=False, rho=0.99)
    conn.get_cvi(data, labels)
    source_ids = conn.get_prototype_ids(10)
    assert len(source_ids) == 4

    value = conn.split(10, 30, source_ids[:2])

    assert value == pytest.approx(1 / 18)
    assert conn.get_prototype_ids(10) == source_ids[2:]
    assert conn.get_prototype_ids(30) == source_ids[:2]
    np.testing.assert_array_equal(conn._cluster_cardinality.asarray(), [2, 2, 2])


def test_fuzzy_merge_compacts_source_before_target():
    data = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5],
         [0.55, 0.55], [1.0, 0.0], [1.0, 1.0]]
    )
    labels = np.asarray([10, 10, 20, 20, 30, 30])
    conn = cvi.CONN(model_type="Fuzzy", normalize_batch=False, rho=0.9)
    conn.get_cvi(data, labels)
    prior_map = dict(conn._artmap.map)
    cadj_before = conn._CADJ.asarray()
    conn_before = conn._CONN.asarray()

    value = conn.merge(target_label=30, source_label=10)

    assert conn._label_map.map == {20: 0, 30: 1}
    assert conn._n_clusters == 2
    assert value == conn.criterion_value
    assert np.isfinite(value)
    for prototype_id, old_label in prior_map.items():
        expected = 0 if old_label == 1 else 1
        assert conn._artmap.map[prototype_id] == expected
    np.testing.assert_array_equal(
        conn._artmap.labels_,
        [conn._artmap.map[idx] for idx in conn._artmap.module_a.labels_],
    )
    np.testing.assert_array_equal(conn._artmap.classes_, [0, 1])
    np.testing.assert_array_equal(conn._CADJ.asarray(), cadj_before)
    np.testing.assert_array_equal(conn._CONN.asarray(), conn_before)

    conn.get_cvi(np.asarray([0.95, 0.95]), 30)
    assert conn._cluster_cardinality.asarray().sum() == conn._n_samples


def test_fuzzy_merge_to_single_cluster_keeps_connectivity_score():
    conn = _fuzzy_operation_fixture()

    value = conn.merge(target_label=20, source_label=10)

    assert conn._label_map.map == {20: 0}
    assert conn.get_prototype_ids(20) == (0, 1, 2, 3)
    np.testing.assert_array_equal(conn._cluster_cardinality.asarray(), [6])
    np.testing.assert_array_equal(conn._INTRA.asarray(), [5 / 6])
    np.testing.assert_array_equal(conn._INTER.asarray(), [[0]])
    assert value == pytest.approx(5 / 6)


def test_fuzzy_split_rejects_invalid_inputs_without_mutation():
    conn = _fuzzy_operation_fixture()
    map_before = dict(conn._artmap.map)
    labels_before = conn._artmap.labels_.copy()
    external_before = dict(conn._label_map.map)
    score_before = conn.criterion_value

    invalid_cases = [
        (10, 30, []),
        (10, 30, [0, 0]),
        (10, 30, [0, 1]),
        (10, 30, [2]),
        (10, 30, [999]),
        (10, 30, [True]),
        (10, 20, [1]),
        (999, 30, [1]),
    ]
    for retained, new, prototypes in invalid_cases:
        with pytest.raises(ValueError):
            conn.split(retained, new, prototypes)
        assert conn._artmap.map == map_before
        np.testing.assert_array_equal(conn._artmap.labels_, labels_before)
        assert conn._label_map.map == external_before
        assert conn.criterion_value == score_before

    with pytest.raises(ValueError):
        conn.merge(10, 999)
    with pytest.raises(ValueError):
        conn.merge(10, 10)
    assert conn._artmap.map == map_before
    assert conn._label_map.map == external_before
