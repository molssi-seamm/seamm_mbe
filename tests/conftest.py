"""Shared fixtures."""

import pytest

import seamm_mbe

from .helpers import Pilot, water_system

# tests/data holds the fixture and the script that made it (provenance only;
# it needs the prototype's tools), not tests
collect_ignore = ["data"]


@pytest.fixture(scope="session")
def pilot():
    return Pilot()


@pytest.fixture(scope="session")
def pilot_system(pilot):
    return water_system(pilot.X, pilot.box)


@pytest.fixture(scope="session")
def pilot_fragments(pilot_system):
    fragments = seamm_mbe.enumerate_fragments(pilot_system)
    seamm_mbe.assign_levels(fragments, {1: True, 2: 3.5})
    return fragments
