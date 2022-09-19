"""
Utilities that are common across all CVI objects.
"""

# Standard library imports
from typing import (
    Callable,
    Union
)
from abc import abstractmethod

# Custom imports
import numpy as np

# --------------------------------------------------------------------------- #
# CLASSES
# --------------------------------------------------------------------------- #


class LabelMap():
    """
    Internal map between labels and the incremental CVI categories.
    """

    def __init__(self):
        self.map = dict()
        return

    def get_internal_label(self, label: int) -> int:
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

    def __init__(self):
        """
        CVI base class initialization method.
        """
        self.label_map = LabelMap()
        self.dim = 0
        self.n_samples = 0
        self.n = []                 # dim
        self.v = np.zeros([0, 0])   # n_clusters x dim
        self.CP = []                # dim
        self.G = np.zeros([0, 0])   # n_clusters x dim
        self.n_clusters = 0
        self.criterion_value = 0.0
        self.is_setup = False

        return

    def setup(self, sample: np.ndarray):
        """
        Common CVI procedure for incremental setup.

        Parameters
        ----------
        data : numpy.ndarray
            Sample vector of features.
        """

        # Infer the dimension as the length of the provided sample
        self.dim = len(sample)

        # Set the sizes of common arrays for consistent appending
        self.v = np.zeros([0, self.dim])
        self.G = np.zeros([0, self.dim])

        # Declare that the CVI is internally setup
        self.is_setup = True

        return

    def setup_batch(self, data: np.ndarray):
        """
        Common CVI procedure for batch setup.

        Parameters
        ----------
        data : np.ndarray
            A batch of samples with some feature dimension.
        """

        # Infer the data dimension and number of samples
        self.n_samples, self.dim = data.shape
        self.is_setup = True

        return

    @abstractmethod
    def param_inc(self, sample: np.ndarray, label: int):
        pass

    @abstractmethod
    def param_batch(self, data: np.ndarray, labels: np.ndarray):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def get_cvi(self, data: np.ndarray, label: Union[int, np.ndarray]) -> float:
        """
        Updates the CVI parameters and then evaluates and returns the criterion value.

        Parameters
        ----------
        data : np.ndarray
            The sample(s) of features used for clustering.
        label : Union[int, np.ndarray]
            The label(s) prescribed to the sample(s) by the clustering algorithm.

        Returns
        -------
        float
            The CVI's criterion value.
        """

        # If we got 1D data, do a quick update
        if (data.ndim == 1):
            self.param_inc(data, label)
            pass
        # Otherwise, we got 2D data and do the correct update
        elif (data.ndim == 2):
            # If we haven't done a batch update yet
            if not self.is_setup:
                # Do a batch update
                self.param_batch(data, label)
            # Otherwise, we are already setup
            else:
                raise ValueError(
                    "Switching from batch to incremental not supported"
                )
                # Do many incremental updates
                # for ix in range(len(label)):
                #     self.param_inc(data[ix, :], label[ix])
        else:
            raise ValueError(
                f"Please provide 1D or 2D numpy array, recieved ndim={data.ndim}"
            )

        # Regardless of path, evaluate and extract the criterion value
        self.evaluate()
        criterion_value = self.criterion_value

        # Return the criterion value
        return criterion_value


# --------------------------------------------------------------------------- #
# DECORATORS
# --------------------------------------------------------------------------- #


def add_docs(other_func: Callable[[], None]) -> Callable[[], None]:
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


def setup_doc():
    """
    Sets up the dimensions of the CVI based on the sample size.

    Parameters
    ----------
    sample : numpy.ndarray
        A sample vector of features.
    """

    pass


# This function documents the shared API for incremental parameter updates
def param_inc_doc():
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
def param_batch_doc():
    """
    Parameters
    ----------
    sample : numpy.ndarray
        A batch of samples; each row is a new sample of features.
    label : numpy.ndarray
        A vector of integer labels, zero-indexed.
    """

    pass


# This function documents the shared API for criterion value evaluation
def evaluate_doc():
    """
    Updates the internal `criterion_value` parameter.
    """

    pass
