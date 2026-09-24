"""Tests for constructing a CVI by its short name."""

import pytest

import src.cvi as cvi


@pytest.mark.parametrize("index_type", cvi.MODULES)
def test_create_cvi_returns_matching_fresh_index(index_type):
    name = index_type.info.name_short
    first = cvi.create_cvi(name)
    second = cvi.create_cvi(name.lower())
    third = cvi.create_cvi(name.upper())

    assert type(first) is index_type
    assert type(second) is index_type
    assert type(third) is index_type
    assert isinstance(first, cvi.CVI)
    assert first is not second


def test_create_cvi_forwards_constructor_options():
    conn = cvi.create_cvi("CONN", model_type="KMeans", kmeans_k=3)

    assert conn.model_type == "KMeans"
    assert conn.kmeans_k == 3


@pytest.mark.parametrize("name", ["missing", "CH "])
def test_create_cvi_rejects_unknown_names(name):
    with pytest.raises(ValueError, match="Available names:.*CH.*cSIL.*rCIP"):
        cvi.create_cvi(name)


def test_create_cvi_requires_string_name():
    with pytest.raises(TypeError, match="CVI name must be a string"):
        cvi.create_cvi(None)
