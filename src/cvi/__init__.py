"""
cvi - A Python library for both incremental and batch cluster validity indices.
"""

# Set the version variable of the package
__version__ = "0.7.2"

# Import CVI modules to the top level
from .modules import (
    CVI,
    CH,
    CONN,
    cSIL,
    DB,
    GD43,
    GD53,
    PS,
    rCIP,
    WB,
    XB,
)

# import compat
from . import compat

# Set these names to be imported
__all__ = [
    "CVI",
    "create_cvi",
    "CH",
    "CONN",
    "cSIL",
    "DB",
    "GD43",
    "GD53",
    "PS",
    "rCIP",
    "WB",
    "XB",
    "compat",
]

# Convenience variable containing all implemented modules
MODULES = [
    CH,
    CONN,
    cSIL,
    DB,
    GD43,
    GD53,
    PS,
    rCIP,
    WB,
    XB,
]

_CVI_BY_NAME = {index.info.name_short.casefold(): index for index in MODULES}


def create_cvi(name: str, **kwargs) -> CVI:
    """Create a new CVI instance from its case-insensitive short name.

    Keyword arguments are passed to the selected index constructor. For
    example, ``create_cvi("CONN", model_type="KMeans")`` forwards
    ``model_type`` to :class:`cvi.CONN`.

    Raises
    ------
    TypeError
        If ``name`` is not a string.
    ValueError
        If ``name`` is not a supported CVI short name.
    """
    if not isinstance(name, str):
        raise TypeError("CVI name must be a string")

    try:
        index_type = _CVI_BY_NAME[name.casefold()]
    except KeyError:
        names = ", ".join(index.info.name_short for index in MODULES)
        raise ValueError(
            f"Unknown CVI {name!r}. Available names: {names}"
        ) from None

    return index_type(**kwargs)
