"""Configuration-specific CVI capability tests."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import src.cvi as cvi
from src.cvi.modules import _base


ALL_OPERATIONS = cvi.CVICapabilities(
    batch=True,
    incremental=True,
    merge=True,
    remove=True,
    split=True,
)


@pytest.mark.parametrize(
    "index_type",
    [index for index in cvi.MODULES if index is not cvi.CONN],
)
def test_numpy_indices_support_all_operations(index_type):
    assert index_type().capabilities == ALL_OPERATIONS


@pytest.mark.parametrize(
    "model_type, incremental",
    [("Fuzzy", True), ("KMeans", False), ("MiniBatchKMeans", False)],
)
def test_conn_capabilities_follow_prototype_model(model_type, incremental):
    assert cvi.CONN(model_type=model_type).capabilities == cvi.CVICapabilities(
        batch=True,
        incremental=incremental,
        merge=True,
        remove=False,
        split=True,
    )


@pytest.mark.parametrize("capacity, incremental", [(None, False), (4, True)])
def test_jax_capabilities_follow_capacity(monkeypatch, capacity, incremental):
    monkeypatch.setattr(
        _base,
        "get_backend",
        lambda backend: SimpleNamespace(name=backend),
    )
    assert cvi.CH(backend="jax", capacity=capacity).capabilities == (
        cvi.CVICapabilities(
            batch=True,
            incremental=incremental,
            merge=False,
            remove=False,
            split=False,
        )
    )


def test_capabilities_describe_configuration_not_initialization():
    index = cvi.CH()
    before = index.capabilities
    index.get_cvi(np.asarray([0.0, 1.0]), 10)
    assert index.capabilities == before


def test_capabilities_are_immutable():
    capabilities = cvi.CONN().capabilities
    with pytest.raises(FrozenInstanceError):
        capabilities.incremental = True
