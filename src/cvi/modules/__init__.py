"""
The :mod:`modules` gathers all CVI implementations.
"""

from ._base import (
    LabelMap,
    CVI,
    add_docs,
    param_inc_doc,
    param_batch_doc,
)

from .CH import CH
from .cSIL import cSIL

__all__ = [
    "LabelMap",
    "CVI",
    "add_docs",
    "param_inc_doc",
    "param_batch_doc",
    "CH",
]
