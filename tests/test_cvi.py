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



# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #

class TestCVI:

    def test_load_data(self):
        # Load the test datasets
        cp = pd.read_csv("data/correct_partition.csv")
        op = pd.read_csv("data/over_partition.csv")
        up = pd.read_csv("data/under_partition.csv")
        lg.info(cp); lg.info(op); lg.info(up)
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