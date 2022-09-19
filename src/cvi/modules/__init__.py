"""
The :mod:`modules` gathers all CVI implementations.
"""

from ._base import (
    LabelMap,
    CVI,
    _add_docs,
    _param_inc_doc,
    _param_batch_doc,
)

from .CH import CH
from .cSIL import cSIL
from .DB import DB
from .GD43 import GD43

__all__ = [
    "LabelMap",
    "CVI",
    "_add_docs",
    "_param_inc_doc",
    "_param_batch_doc",
    "CH",
    "cSIL",
    "DB",
    "GD43",
]
