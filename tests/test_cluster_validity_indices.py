"""
    test_cluster_validity_indices.py

Tests the cluster_validity_indices package.
"""

# --------------------------------------------------------------------------- #
# STANDARD IMPORTS
# --------------------------------------------------------------------------- #

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# CUSTOM IMPORTS
# --------------------------------------------------------------------------- #

import pytest

# --------------------------------------------------------------------------- #
# LOCAL IMPORTS
# --------------------------------------------------------------------------- #
print(f"\nTesting path is: {os.getcwd()}")
import src.cluster_validity_indices as cvi
# import ..

# --------------------------------------------------------------------------- #
# TESTS
# --------------------------------------------------------------------------- #

class TestCVI:

    def test_opts(self):
        my_opts = cvi.CVIOpts()
        print(my_opts)

    def test_cvi(self):
        my_cvi =cvi.cvi.CH(3)
        print(my_cvi)