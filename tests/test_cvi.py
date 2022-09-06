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

# --------------------------------------------------------------------------- #
# LOCAL IMPORTS
# --------------------------------------------------------------------------- #
print(f"\nTesting path is: {os.getcwd()}")
import src.cvi as cvi
# import ..

# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #

class TestCVI:

    def test_opts(self):
        my_opts = cvi.CVIOpts()
        lg.info(my_opts)
        return

    def test_cvi(self):
        my_cvi =cvi.cvi.CH()
        lg.info(my_cvi)
        return

    def test_all_cvis(sefl):
        my_cvi = cvi.CH()
        sample = np.ones(3)
        my_cvi.setup(sample)
        lg.info("asdf")
        lg.info(my_cvi)
        return