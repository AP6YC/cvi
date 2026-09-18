"""Internal numerical backends; optional dependencies are loaded on demand."""

from .numpy import NumpyBackend


def get_backend(name):
    """Resolve an explicit backend without importing Numba for NumPy users."""
    if name == "numpy":
        return NumpyBackend()
    if name == "numba":
        try:
            from .numba import NumbaBackend
        except ModuleNotFoundError as error:
            if error.name != "numba":
                raise
            raise ImportError(
                "The numba backend requires the optional dependency; "
                'install it with pip install "cvi[numba]".'
            ) from error
        return NumbaBackend()
    raise ValueError("backend must be 'numpy' or 'numba'")
