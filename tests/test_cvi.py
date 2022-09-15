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
from typing import List, Dict


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
# TODO: this is a hack; refactor modules so that this is at the top
# from src.cvi import CVI

print(f"\nTesting path is: {os.getcwd()}")


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
    correct = (
        pd.read_csv(data_path.joinpath("correct_partition.csv"))
        .sample(frac=p)
        .sort_index()
    )
    over = (
        pd.read_csv(data_path.joinpath("over_partition.csv"))
        .sample(frac=p)
        .sort_index()
    )
    under = (
        pd.read_csv(data_path.joinpath("under_partition.csv"))
        .sample(frac=p)
        .sort_index()
    )

    # Coerce the dataframe as two numpy arrays each for ease
    correct_samples = correct.to_numpy(dtype=float)[:, 0:2]
    correct_labels = correct.to_numpy(dtype=int)[:, 2] - 1
    over_samples = over.to_numpy(dtype=float)[:, 0:2]
    over_labels = over.to_numpy(dtype=int)[:, 2] - 1
    under_samples = under.to_numpy(dtype=float)[:, 0:2]
    under_labels = under.to_numpy(dtype=int)[:, 2] - 1

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
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------- #


def get_cvis() -> List[cvi.CVI]:
    """
    Returns a list of constructed CVI modules.
    """

    cvis = [
        cvi.DB(),
        cvi.cSIL(),
        cvi.CH(),
    ]

    return cvis


def log_data(local_data: Dict) -> None:
    """
    Info-logs aspects of the passed data dictionary for diagnosis.
    """

    # lg.info(local_data.describe())
    lg.info(
        f"Samples: type {type(local_data['samples'])}, "
        f"shape {local_data['samples'].shape}"
    )
    lg.info(
        f"Labels: type {type(local_data['labels'])}, "
        f"shape {local_data['labels'].shape}"
    )
    return


def get_sample(local_data: Dict, index: int) -> tuple:
    """
    Grabs a sample and label from the data dictionary at the provided index.
    """

    sample = local_data["samples"][index, :]
    label = local_data["labels"][index]

    return sample, label


# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #


class TestCVI:

    def test_load_data(self, data: TestData) -> None:
        """
        Test loading the partitioning data.
        """

        lg.info("--- TESTING DATA LOADING ---")
        lg.info(f"Data location: {id(data)}")

        for value in data.datasets.values():
            log_data(value)

        return

    def test_loading_again(self, data: TestData) -> None:
        """
        Tests loading the data again to verify the identity of the data dictionary.
        """

        lg.info("--- TESTING LOADING AGAIN TO VERIFY DATA SINGLETON ---")
        log_data(data.datasets["correct"])
        lg.info(f"Data location: {id(data)}")

        return

    def test_icvis(self, data: TestData) -> None:
        """
        Test the functionality all of the icvis.
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
