"""Focused tests for the generalized Dunn validity indices."""

import warnings

import numpy as np
import pytest

import src.cvi as cvi


GENERALIZED_DUNN_INDICES = [cvi.GD43, cvi.GD53]


@pytest.mark.parametrize("cvi_type", GENERALIZED_DUNN_INDICES)
@pytest.mark.parametrize(
    ("samples", "labels"),
    [
        (
            np.asarray([[0.0], [2.0]]),
            np.asarray([10, 20]),
        ),
        (
            np.asarray([[0.0], [0.0], [2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
    ],
    ids=["singleton-clusters", "identical-members"],
)
def test_batch_zero_dispersion_returns_nan(cvi_type, samples, labels):
    """Zero dispersion warns and returns NaN."""

    local_cvi = cvi_type()

    message = f"{cvi_type.__name__} is undefined for the supplied batch"
    with pytest.warns(RuntimeWarning, match=message):
        result = local_cvi.get_cvi(samples, labels)

    assert np.isnan(result)
    assert np.isnan(local_cvi.criterion_value)
    assert local_cvi._intra == 0.0


@pytest.mark.parametrize("cvi_type", GENERALIZED_DUNN_INDICES)
def test_incremental_zero_dispersion_returns_nan(cvi_type):
    """Singleton clusters remain quietly undefined until dispersion exists."""

    local_cvi = cvi_type()
    assert np.isnan(local_cvi.get_cvi(np.asarray([0.0]), 10))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.get_cvi(np.asarray([2.0]), 20)

    assert np.isnan(result)
    assert local_cvi._intra == 0.0

    result = local_cvi.get_cvi(np.asarray([1.0]), 10)
    assert np.isfinite(result)


@pytest.mark.parametrize("cvi_type", GENERALIZED_DUNN_INDICES)
def test_remove_to_zero_dispersion_returns_nan(cvi_type):
    """An operation returns NaN when it makes the index undefined."""

    samples = np.asarray([[0.0], [1.0], [3.0]])
    labels = np.asarray([10, 10, 20])
    local_cvi = cvi_type()
    assert np.isfinite(local_cvi.get_cvi(samples, labels))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.remove(np.asarray([1.0]), 10)

    assert np.isnan(result)
    assert local_cvi._intra == 0.0


@pytest.mark.parametrize(
    ("cvi_type", "samples", "labels"),
    [
        (
            cvi.GD43,
            np.asarray([[-1.0], [1.0], [0.0]]),
            np.asarray([10, 10, 20]),
        ),
        (
            cvi.GD53,
            np.asarray([[-1.0], [1.0], [3.0], [4.0]]),
            np.asarray([10, 10, 20, 30]),
        ),
    ],
)
def test_zero_score_can_be_defined(cvi_type, samples, labels):
    """A computed zero is distinguishable from undefined NaN."""

    local_cvi = cvi_type()
    assert local_cvi.get_cvi(samples, labels) == 0.0
    assert local_cvi._intra > 0.0
