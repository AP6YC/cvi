"""
    test_cvi.py

Tests the cvi package.
"""

# --------------------------------------------------------------------------- #
# STANDARD IMPORTS
# --------------------------------------------------------------------------- #

import os
from pathlib import Path
import logging as lg
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Tuple,
)

# --------------------------------------------------------------------------- #
# CUSTOM IMPORTS
# --------------------------------------------------------------------------- #

import pytest
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# LOCAL IMPORTS
# --------------------------------------------------------------------------- #

import src.cvi as cvi

print(f"\nTesting path is: {os.getcwd()}")


# --------------------------------------------------------------------------- #
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------- #


def get_cvis() -> List[cvi.CVI]:
    """
    Returns a list of constructed CVI modules.

    Returns
    -------
    List[cvi.CVI]
        A list of constructed CVI objects for fresh use.
    """

    # Construct a list of CVI objects
    cvis = [
        cvi.DB(),
        cvi.cSIL(),
        cvi.CH(),
    ]

    return cvis


def log_data(local_data: Dict) -> None:
    """
    Info-logs aspects of the passed data dictionary for diagnosis.

    Parameters
    ----------
    local_data : Dict
        A dictionary containing arrays of samples and labels.
    """

    # Log the type, shape, and number of samples and labels
    lg.info(
        f"Samples: type {type(local_data['samples'])}, "
        f"shape {local_data['samples'].shape}"
    )
    lg.info(
        f"Labels: type {type(local_data['labels'])}, "
        f"shape {local_data['labels'].shape}"
    )
    return


def get_sample(local_data: Dict, index: int) -> Tuple[np.ndarray, int]:
    """
    Grabs a sample and label from the data dictionary at the provided index.

    Parameters
    ----------
    local_data : Dict
        Dictionary containing an array of samples and vector of labels.
    index : int
        The index to load the sample at.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple of sample features and the integer label prescribed to the sample.

    """

    # Grab a sample and label at the index
    sample = local_data["samples"][index, :]
    label = local_data["labels"][index]

    return sample, label


def load_pd_csv(data_path: Path, frac: float) -> pd.DataFrame:
    """
    Loads a csv file using pandas, subsampling the data at the given fraction while preservign order.

    Parameters
    ----------
    data_path : Path
        The pathlib.Path where the data .csv file is.
    frac : float
        The data subsampling fraction within (0, 1].

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the subsampled data.
    """

    # Load the data as a pandas array, subsample, and preserve order by the index
    local_data = (
        pd.read_csv(data_path)
        .sample(frac=frac)
        .sort_index()
    )

    return local_data


def split_data_columns(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a pandas DataFrame into numpy arrays of samples and labels, assuming the last column is labels.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the samples and their corresponding labels.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple of numpy arrays containing the separate samples and their corresponding labels.
    """

    # Index to before the last index, correct for python 0-indexing.
    samples = df.to_numpy(dtype=float)[:, :-1]
    labels = df.to_numpy(dtype=int)[:, -1] - 1

    return samples, labels

# --------------------------------------------------------------------------- #
# DATACLASSES
# --------------------------------------------------------------------------- #


@dataclass
class TestData():
    """
    A container dataclass for test data.
    """

    # The test dataset dictionary
    datasets: Dict

    # Tells pytest that this is not a test class
    __test__ = False

    def count(self, dataset: str) -> int:
        """
        Returns the number of samples in a dataset entry.

        Parameters
        ----------
        dataset : str
            The key corresponding to which dataset you wish to get a count of.

        Returns
        -------
        int
            The number of samples in self.datasets[dataset].
        """

        return len(self.datasets[dataset]["labels"])


# --------------------------------------------------------------------------- #
# FIXTURES
# --------------------------------------------------------------------------- #


# Set the fixture scope to the testing session to load the data once
@pytest.fixture(scope="session")
def data() -> TestData:
    """
    Data loading test fixture.

    This fixture is run once for the entire pytest session.
    """

    p = 0.1
    lg.info("LOADING DATA")

    data_path = Path("tests", "data")

    # Load the test datasets
    correct = load_pd_csv(data_path.joinpath("correct_partition.csv"), p)
    over = load_pd_csv(data_path.joinpath("over_partition.csv"), p)
    under = load_pd_csv(data_path.joinpath("under_partition.csv"), p)

    # Coerce the dataframe as two numpy arrays each for ease
    correct_samples, correct_labels = split_data_columns(correct)
    over_samples, over_labels = split_data_columns(over)
    under_samples, under_labels = split_data_columns(under)

    # Construct the dataset dictionary
    data_dict = {
        "correct": {
            "samples": correct_samples,
            "labels": correct_labels,
        },
        "over": {
            "samples": over_samples,
            "labels": over_labels,
        },
        "under": {
            "samples": under_samples,
            "labels": under_labels,
        },
    }

    # Instantiate and return the TestData object
    return TestData(data_dict)


# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #


class TestCVI:

    def test_load_data(self, data: TestData) -> None:
        """
        Test loading the partitioning data.

        Parameters
        ----------
        data : TestData
            The data loaded as a pytest fixture.
        """

        lg.info("--- TESTING DATA LOADING ---")
        lg.info(f"Data location: {id(data)}")

        for value in data.datasets.values():
            log_data(value)

        return

    def test_loading_again(self, data: TestData) -> None:
        """
        Tests loading the data again to verify the identity of the data dictionary.

        Parameters
        ----------
        data : TestData
            The data loaded as a pytest fixture.
        """

        lg.info("--- TESTING LOADING AGAIN TO VERIFY DATA SINGLETON ---")
        log_data(data.datasets["correct"])
        lg.info(f"Data location: {id(data)}")

        return

    def test_icvis(self, data: TestData) -> None:
        """
        Test the functionality all of the icvis.

        Parameters
        ----------
        data : TestData
            The data loaded as a pytest fixture.
        """

        lg.info("--- TESTING ALL ICVIS ---")

        tolerance = 1e-1

        for key, local_data in data.datasets.items():
            lg.info(f"Testing data: {key}")
            # Incremental
            i_cvis = get_cvis()
            for local_cvi in i_cvis:
                for ix in range(data.count(key)):
                    # Grab a sample and label
                    sample, label = get_sample(local_data, ix)
                    local_cvi.param_inc(sample, label)
                    local_cvi.evaluate()
            # Batch
            b_cvis = get_cvis()
            for local_cvi in b_cvis:
                local_cvi.param_batch(
                    local_data["samples"],
                    local_data["labels"]
                )
                local_cvi.evaluate()

            # Test equivalence between batch and incremental results
            for i in range(len(i_cvis)):
                assert (
                    (i_cvis[i].criterion_value - b_cvis[i].criterion_value)
                    < tolerance
                )
                lg.info(
                    f"I: {i_cvis[i].criterion_value},"
                    f"B: {b_cvis[i].criterion_value}"
                )

        return

    def test_setup_icvi(self) -> None:
        """
        Test running the setup method on ICVIs.
        """

        # Create the cvi/icvi module
        my_cvi = cvi.modules.CH()

        lg.info("Before setup")
        lg.info(my_cvi)

        sample = np.ones(3)
        my_cvi.setup(sample)

        lg.info("After setup")
        lg.info(my_cvi)

        return
