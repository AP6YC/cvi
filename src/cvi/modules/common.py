"""
Utilities that are common across all CVI objects.
"""

# Standard library imports
from typing import Callable

# Custom imports
import numpy as np

# --------------------------------------------------------------------------- #
# CLASSES
# --------------------------------------------------------------------------- #

class LabelMap():
    """
    Internal map between labels and the incremental CVI categories.
    """

    def __init__(self) -> None:
        self.map = dict()
        return

    def get_internal_label(self, label:int) -> int:
        """
        Gets the internal label and updates the label map if the label is new.
        """

        # Initialize the internal label
        internal_label = None

        # If the label is in the map, return that
        if label in self.map:
            internal_label = self.map[label]
        # Otherwise, create an incremented new label and return that
        else:
            # Correct for python zero-indexing by not including the +1
            internal_label = len(self.map.items())
            self.map[label] = internal_label

        return internal_label


class CVI():
    """
    Superclass containing elements shared between all CVIs.
    """

    def __init__(self) -> None:
        """
        CVI base class initialization method.
        """
        self.label_map = LabelMap()
        self.dim = 0
        self.n_samples = 0
        self.mu = np.empty([0])     # dim
        self.n = []                 # dim
        self.v = np.empty([0, 0])   # n_clusters x dim
        self.CP = []                # dim
        self.SEP = []               # dim
        self.G = np.empty([0, 0])   # n_clusters x dim
        self.BGSS = 0.0
        self.WGSS = 0.0
        self.n_clusters = 0
        self.criterion_value = 0.0

        return

    def setup(self, sample:np.ndarray) -> None:
        """
        Sets up the dimensions of the CVI based on the sample size.

        Parameters
        ----------
        sample : numpy.ndarray
            A sample vector of features.
        """
        self.dim = len(sample)
        # self.v = np.empty([dim, 0])
        # self.G = np.empty([dim, 0])

        self.mu = np.empty([self.dim])
        # self.n = []
        self.v = np.empty([0, self.dim])
        # self.CP = np.empty([self.dim])
        # self.CP = []
        self.SEP = np.empty([self.dim])
        self.G = np.empty([0, self.dim])

        return

# --------------------------------------------------------------------------- #
# DECORATORS
# --------------------------------------------------------------------------- #

def add_docs(other_func:Callable[[], None]) -> Callable[[], None]:
    """
    A decorator for appending the docstring of one function to another.

    Parameters
    ----------
    other_func : Callable[[], None]
        The other function whose docstring you want to append to the decorated function.
    """

    def dec(func):
        func.__doc__ = func.__doc__ + other_func.__doc__
        return func

    return dec

# --------------------------------------------------------------------------- #
# DOCSTRING FUNCTIONS
# --------------------------------------------------------------------------- #

# This function documents the shared API for incremental parameter updates
def param_inc_doc() -> None:
    """
    Parameters
    ----------
    sample : numpy.ndarray
        A sample row vector of features.
    label : int
        An integer label for the cluster, zero-indexed.
    """

    pass

# This function documents the shared API for batch parameter updates
def param_batch() -> None:
    """
    Parameters
    ----------
    sample : numpy.ndarray
        A batch of samples; each row is a new sample of features.
    label : numpy.ndarray
        A vector of integer labels, zero-indexed.
    """

    pass
