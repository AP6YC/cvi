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
# import inspect

# --------------------------------------------------------------------------- #
# CUSTOM IMPORTS
# --------------------------------------------------------------------------- #

import pytest
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# LOCAL IMPORTS
# --------------------------------------------------------------------------- #

print(f"\nTesting path is: {os.getcwd()}")
import src.cvi as cvi
# import ..

# --------------------------------------------------------------------------- #
# FIXTURES
# --------------------------------------------------------------------------- #

# Set the fixture scope to the testing session to load the data once
@pytest.fixture(scope="session")
def data():
    """
    Data loading test fixture.

    This fixture is run once for the entire pytest session.
    """

    p = 0.1
    lg.info("LOADING DATA")

    data_path = Path("tests", "data")
    # Load the test datasets
    # correct = pd.read_csv(data_path.joinpath("correct_partition.csv")).sample(frac=p).convert_dtypes()
    # over = pd.read_csv(data_path.joinpath("over_partition.csv")).sample(frac=p).convert_dtypes()
    # under = pd.read_csv(data_path.joinpath("under_partition.csv")).sample(frac=p).convert_dtypes()
    correct = pd.read_csv(data_path.joinpath("correct_partition.csv"))
    over = pd.read_csv(data_path.joinpath("over_partition.csv"))
    under = pd.read_csv(data_path.joinpath("under_partition.csv"))

    # Coerce the dataframe as two numpy arrays each for ease
    correct_samples = correct.to_numpy(dtype=float)[:, 0:2]
    correct_labels = correct.to_numpy(dtype=int)[:, 2] - 1
    over_samples = over.to_numpy(dtype=float)[:, 0:2]
    over_labels = over.to_numpy(dtype=int)[:, 2] - 1
    under_samples = under.to_numpy(dtype=float)[:, 0:2]
    under_labels = under.to_numpy(dtype=int)[:, 2] - 1

    # Construct the dataset dictionary
    data_dict = {
        "datasets": {
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
        },
        "counts": {
            "correct": len(correct.index),
            "over": len(over.index),
            "under": len(under.index),
        },
    }

    return data_dict

# @pytest.fixture()
def get_cvis():
    """
    Returns a list of constructed CVI modules.
    """
    cvis = [
        cvi.modules.CH(),
    ]

    return cvis

def log_data(local_data):
    """
    Info-logs aspects of the passed data dictionary for diagnosis.
    """
    # lg.info(local_data.describe())
    lg.info(
        f"Samples: type {type(local_data['samples'])}, shape {local_data['samples'].shape}"
    )
    lg.info(
        f"Labels: type {type(local_data['labels'])}, shape {local_data['labels'].shape}"
    )
    return

def get_sample(local_data, index:int) -> tuple:
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

    def test_load_data(self, data):
        """Test loading the partitioning data.
        """

        lg.info("--- TESTING DATA LOADING ---")
        lg.info(f"Data location: {id(data)}")

        for value in data["datasets"].values():
            log_data(value)

        return

    def test_loading_again(self, data):
        """
        Tests loading the data again to verify the identity of the data dictionary.
        """

        lg.info("--- TESTING LOADING AGAIN TO VERIFY DATA SINGLETON ---")
        log_data(data["datasets"]["correct"])
        lg.info(f"Data location: {id(data)}")

        return

    def test_icvis(self, data):
        """
        Test the functionality all of the icvis.
        """

        lg.info("--- TESTING ALL ICVIS ---")

        tolerance = 1e-1

        # n_cvis = len(cvis)

        for key, local_data in data["datasets"].items():
            lg.info(f"Testing data: {key}")
            cvis = get_cvis()
            for local_cvi in cvis:
                lg.info(local_cvi)
                for ix in range(data["counts"][key]):
                    # lg.info(f"--- ITERATION {ix} ---")
                    # Grab a sample and label
                    sample, label = get_sample(local_data, ix)
                    # lg.info(f"Sample: {type(sample)}, {sample}, label: {type(label)}, {label}")
                    local_cvi.param_inc(sample, label)
                    local_cvi.evaluate()

        return

    def test_opts(self):
        my_opts = cvi.CVIOpts()
        lg.info(my_opts)

        return

    def test_setup_icvi(self):
        """Test running the setup method on ICVIs.
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