"""Backend coverage shared by numerical regression tests."""

import importlib

import pytest


@pytest.fixture(params=["numpy", "numba"])
def backend(request):
    if request.param == "numba":
        pytest.importorskip("numba")
    return request.param


@pytest.fixture
def jax_runtime():
    jax = pytest.importorskip("jax")
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield jax, importlib.import_module("src.cvi.jax")
    finally:
        jax.config.update("jax_enable_x64", previous)
