"""Assembling labels: energy offsets per type, the molecular virial, clusters."""

import numpy as np
import pytest

import seamm_mbe


def li_water_cell():
    """One Li+ and two waters in a 20 Å cell."""
    symbols = ["Li", "O", "H", "H", "O", "H", "H"]
    xyz = [
        [10.0, 10, 10],
        [12.0, 10, 10],
        [12.6, 10.8, 10],
        [12.6, 9.2, 10],
        [8.0, 10, 10],
        [7.4, 10.8, 10],
        [7.4, 9.2, 10],
    ]
    bonds = [(1, 2), (1, 3), (4, 5), (4, 6)]
    return seamm_mbe.System(symbols, xyz, np.eye(3) * 20.0, bonds=bonds)


def monomers_only(system, rng, zero_net=True):
    rules = seamm_mbe.SelectionRules(max_order=1, cutoffs={})
    fragments = seamm_mbe.enumerate_fragments(system, rules)
    seamm_mbe.assign_levels(fragments)
    high, low = {}, {}
    for f in fragments:
        forces = rng.normal(size=(f.n_atoms, 3))
        if zero_net:
            forces -= forces.mean(axis=0)
        high[f.name] = (rng.normal(), forces)
        low[f.name] = (0.0, np.zeros((f.n_atoms, 3)))
    return seamm_mbe.mbe_correction(fragments, high, molecular=low)


def test_offsets_for_a_two_type_mixture():
    system = li_water_cell()
    correction = monomers_only(system, np.random.default_rng(1))
    cell = seamm_mbe.CellTerm("low", -50.0, np.zeros((7, 3)), np.zeros((3, 3)))
    labels = seamm_mbe.assemble(
        system, correction, [cell], offsets={"water": 2.0, "Li+": 10.0}
    )
    assert labels.energy == pytest.approx(-50.0 + correction.energy)
    assert labels.reference_energy == pytest.approx(labels.energy + 2 * 2.0 + 10.0)
    with pytest.raises(ValueError, match="Li\\+"):
        seamm_mbe.assemble(system, correction, [cell], offsets={"water": 2.0})


def test_monomer_increments_drop_out_of_the_molecular_virial():
    """A monomer increment with no net force changes the atomic pressure but
    not the molecular one."""
    system = li_water_cell()
    correction = monomers_only(system, np.random.default_rng(2))
    cell = seamm_mbe.CellTerm("low", 0.0, np.zeros((7, 3)), np.zeros((3, 3)))
    labels = seamm_mbe.assemble(system, correction, [cell])
    assert abs(labels.pressure) > 1.0
    assert labels.molecular_pressure == pytest.approx(0.0, abs=1e-9)
    w_intra = seamm_mbe.intramolecular_virial(system, labels.forces)
    assert np.allclose(w_intra, correction.virial)


def test_cell_terms_must_match():
    system = li_water_cell()
    correction = monomers_only(system, np.random.default_rng(3))
    with pytest.raises(ValueError, match="forces of shape"):
        seamm_mbe.assemble(
            system, correction, [seamm_mbe.CellTerm("x", 0, np.zeros((6, 3)))]
        )
    with pytest.raises(ValueError, match="no virial"):
        seamm_mbe.assemble(
            system, correction, [seamm_mbe.CellTerm("x", 0, np.zeros((7, 3)))]
        )


def test_cluster_labels_have_no_stress():
    system = seamm_mbe.System(
        ["O", "H", "H"],
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
        bonds=[(0, 1), (0, 2)],
    )
    correction = monomers_only(system, np.random.default_rng(4))
    labels = seamm_mbe.assemble(
        system, correction, [seamm_mbe.CellTerm("low", 1.0, np.zeros((3, 3)))]
    )
    assert labels.virial is None and labels.stress is None
    assert labels.pressure is None and labels.molecular_pressure is None
    assert labels.energy == pytest.approx(1.0 + correction.energy)
