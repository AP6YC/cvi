"""
cvi - A Python library for both incremental and batch cluster validity indices.
"""

# Set the version variable of the package
__version__ = "0.1.1"

# Import CVI modules to the top level
from .modules import (
    CVI,
    CH,
    cSIL,
    DB,
)

# Set these names to be imported
__all__ = [
    "CVI",
    "CH",
    "cSIL",
    "DB",
]

# Convenience variable containing all implemented modules
MODULES = [
    CH,
    cSIL,
    DB,
]