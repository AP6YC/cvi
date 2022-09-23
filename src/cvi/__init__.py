"""
cvi - A Python library for both incremental and batch cluster validity indices.
"""

# Set the version variable of the package
__version__ = "0.3.0"

# Import CVI modules to the top level
from .modules import (
    CVI,
    CH,
    cSIL,
    DB,
    GD43,
    GD53,
    rCIP,
    WB,
    XB,
)

# import compat
from . import compat

# Set these names to be imported
__all__ = [
    "CVI",
    "CH",
    "cSIL",
    "DB",
    "GD43",
    "GD53",
    "rCIP",
    "WB",
    "XB",
    "compat",
]

# Convenience variable containing all implemented modules
MODULES = [
    CH,
    cSIL,
    DB,
    GD43,
    GD53,
    rCIP,
    WB,
    XB,
]
