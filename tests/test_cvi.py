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
    """Data loading test fixture.
    """

    p = 0.1
    lg.info("LOADING DATA")
    # Load the test datasets
    correct = pd.read_csv("tests/data/correct_partition.csv").sample(frac=p).convert_dtypes()
    over = pd.read_csv("tests/data/over_partition.csv").sample(frac=p).convert_dtypes()
    under = pd.read_csv("tests/data/under_partition.csv").sample(frac=p).convert_dtypes()

    # Construct the dataset dictionary
    data_dict = {
        "datasets": {
            "correct": correct,
            "over": over,
            "under": under,
        },
        "counts": {
            "correct": len(correct.index),
            "over": len(over.index),
            "under": len(under.index),
        },
    }

    return data_dict

@pytest.fixture()
def cvis():

    cvis = [
        cvi.modules.CH,
    ]

    return cvis

def log_data(local_data):
    lg.info(local_data.describe())
    return


# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #

class TestCVI:

    def test_load_data(self, data):
        """Test loading the partitioning data.
        """

        lg.info(f"Data location: {id(data)}")

        for value in data["datasets"].values():
            log_data(value)

        return

    def test_loading_again(self, data):
        """
        """
        log_data(data["datasets"]["correct"])
        lg.info(f"Data location: {id(data)}")
        return

    def test_icvis(self, data, cvis):
        """Test the functionality all of the icvis.
        """

        tolerance = 1e-1

        n_cvis = len(cvis)

        for key, local_data in data["datasets"].items():
            lg.info(f"Testing data: {key}")
            for local_cvi in cvis:
                # for ix in range(data["counts"][key]):
                for index, row in local_data.iterrows():
                    sample = row[0:1]
                    label = row[2]
                    # sample =

                # for ix

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