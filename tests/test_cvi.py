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

@pytest.fixture
def data():
    correct = pd.read_csv("tests/data/correct_partition.csv")
    over = pd.read_csv("tests/data/over_partition.csv")
    under = pd.read_csv("tests/data/under_partition.csv")
    data_dict = {"correct": correct, "over": over, "under": under}
    return data_dict

# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #

class TestCVI:

    def test_load_data(self, data):
        # Load the test datasets
        # cp = pd.read_csv("data/correct_partition.csv")
        # op = pd.read_csv("data/over_partition.csv")
        # up = pd.read_csv("data/under_partition.csv")
        lg.info(data["correct"]); lg.info(data["over"]); lg.info(data["under"])
        return

    def test_opts(self):
        my_opts = cvi.CVIOpts()
        lg.info(my_opts)
        return

    def test_cvi(sefl):
        my_cvi = cvi.modules.CH()
        lg.info(my_cvi)
        sample = np.ones(3)
        my_cvi.setup(sample)
        lg.info("asdf")
        lg.info(my_cvi)
        return