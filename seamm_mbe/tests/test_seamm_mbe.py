"""
Unit and regression test for the seamm_mbe package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import seamm_mbe


def test_seamm_mbe_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seamm_mbe" in sys.modules
