"""Tests for the common CVI undefined-value contract."""

import warnings

import numpy as np
import pytest

import src.cvi as cvi


CVIS_NOT_CONN = [cvi_type for cvi_type in cvi.MODULES if cvi_type is not cvi.CONN]


@pytest.mark.parametrize("cvi_type", cvi.MODULES)
def test_criterion_value_starts_as_nan(cvi_type):
    """Every current CVI starts with an undefined NaN result."""

    local_cvi = cvi_type()

    assert np.isnan(local_cvi.criterion_value)
    assert not hasattr(local_cvi, "is_defined")


@pytest.mark.parametrize("cvi_type", CVIS_NOT_CONN)
def test_incremental_result_becomes_finite_when_formula_is_defined(cvi_type):
    """A normal stream remains undefined until its second cluster appears."""

    local_cvi = cvi_type()

    assert np.isnan(local_cvi.get_cvi(np.asarray([0.0]), 10))

    assert np.isnan(local_cvi.get_cvi(np.asarray([1.0]), 10))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.get_cvi(np.asarray([3.0]), 20)

    assert np.isfinite(result)


@pytest.mark.parametrize(
    ("cvi_type", "samples", "labels"),
    [
        (
            cvi.CH,
            np.asarray([[0.0], [2.0]]),
            np.asarray([10, 20]),
        ),
        (
            cvi.WB,
            np.asarray([[-1.0], [1.0], [-2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.DB,
            np.asarray([[-1.0], [1.0], [-2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.XB,
            np.asarray([[-1.0], [1.0], [-2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.GD43,
            np.asarray([[0.0], [0.0], [2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.GD53,
            np.asarray([[0.0], [0.0], [2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.PS,
            np.asarray([[-1.0], [1.0], [-2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
    ],
    ids=["CH", "WB", "DB", "XB", "GD43", "GD53", "PS"],
)
def test_undefined_batch_returns_nan_with_warning(
    cvi_type,
    samples,
    labels,
):
    """Formula-specific undefined batches warn and return NaN."""

    local_cvi = cvi_type()

    message = f"{cvi_type.__name__} is undefined for the supplied batch"
    with pytest.warns(RuntimeWarning, match=message):
        result = local_cvi.get_cvi(samples, labels)

    assert np.isnan(result)
    assert np.isnan(local_cvi.criterion_value)


@pytest.mark.parametrize(
    ("cvi_type", "samples", "labels"),
    [
        (
            cvi.CH,
            np.asarray([[-1.0], [1.0], [-2.0], [2.0]]),
            np.asarray([10, 10, 20, 20]),
        ),
        (
            cvi.WB,
            np.asarray([[0.0], [2.0]]),
            np.asarray([10, 20]),
        ),
        (
            cvi.DB,
            np.asarray([[0.0], [2.0]]),
            np.asarray([10, 20]),
        ),
        (
            cvi.XB,
            np.asarray([[0.0], [2.0]]),
            np.asarray([10, 20]),
        ),
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
    ids=["CH", "WB", "DB", "XB", "GD43", "GD53"],
)
def test_computed_zero_can_be_defined(cvi_type, samples, labels):
    """A valid computed zero remains distinct from undefined NaN."""

    local_cvi = cvi_type()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.get_cvi(samples, labels)

    assert result == 0.0


def test_csil_zero_ties_are_defined_neutral_terms():
    """The published silhouette tie convention avoids a local zero divide."""

    local_cvi = cvi.cSIL()
    samples = np.zeros((4, 1))
    labels = np.asarray([10, 10, 20, 20])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.get_cvi(samples, labels)

    assert result == 0.0
    np.testing.assert_array_equal(local_cvi._sil_coefs, [0.0, 0.0])


def test_rcip_is_finite_with_two_coincident_clusters():
    """rCIP's covariance regularization keeps this state defined."""

    local_cvi = cvi.rCIP()
    samples = np.zeros((4, 1))
    labels = np.asarray([10, 10, 20, 20])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = local_cvi.get_cvi(samples, labels)

    assert np.isfinite(result)


def test_conn_becomes_defined_after_second_art_category():
    """Incremental CONN is unavailable until its second prototype exists."""

    local_cvi = cvi.CONN(model_type="Fuzzy", normalize_batch=False)

    assert np.isnan(local_cvi.get_cvi(np.asarray([0.0]), 10))

    result = local_cvi.get_cvi(np.asarray([1.0]), 20)
    assert len(local_cvi._artmap.module_a.W) == 2
    assert np.isfinite(result)
