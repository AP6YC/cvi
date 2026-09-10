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


def test_conn_is_public_and_defaults_to_minibatch_kmeans():
    assert cvi.modules.CONN is cvi.CONN
    assert cvi.CONN in cvi.MODULES
    assert cvi.CONN().model_type == "MiniBatchKMeans"
    assert cvi.CONN.info.name_short == "CONN"
    assert cvi.CONN.info.index_min == 0.0
    assert cvi.CONN.info.index_max == 1.0
    assert cvi.CONN.info.optimality == "max"
