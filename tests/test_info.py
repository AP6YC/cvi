"""Tests for public CVI metadata."""

from dataclasses import FrozenInstanceError, fields

import pytest
import src.cvi as cvi


def test_cvi_info_contains_static_type_metadata():
    assert tuple(field.name for field in fields(cvi.CH.info)) == (
        "name",
        "name_short",
        "index_min",
        "index_max",
        "optimality",
        "remove",
        "backends",
    )


def test_remove_capability_distinguishes_conn():
    assert not cvi.CONN.info.remove
    assert all(index.info.remove for index in cvi.MODULES if index is not cvi.CONN)


def test_cvi_info_is_immutable():
    with pytest.raises(FrozenInstanceError):
        cvi.CH.info.name = "Changed"
