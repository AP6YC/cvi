"""
Compatability module for previous CVI versions.
"""

from .v0 import (
    iCVI,
    iDB,
    iSIL,
    iGD43,
    iGD53,
    iCH,
    iPS,
    iXB,
)

__all__ = [
    "iCVI",
    "iDB",
    "iSIL",
    "iGD43",
    "iGD53",
    "iCH",
    "iPS",
    "iXB",
]

MODULES = [
    iDB,
    iSIL,
    iGD43,
    iGD53,
    iCH,
    iPS,
    iXB,
]
